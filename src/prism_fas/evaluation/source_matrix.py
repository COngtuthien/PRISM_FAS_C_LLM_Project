"""The v1.5 source-only experiment matrix (§18, §19, §15.4).

§18 fixes the matrix and §18.3 fixes the replication policy, so this module
transcribes rather than decides. It exists because the inherited
`experiment_matrix` module materializes the Version-B M10 matrix, which is a
different set of rows answering different questions; reusing it for C8 would
quietly substitute one experiment plan for another.

What §18 asks for, and what `plan_source_matrix` produces:

* **Track G, three arms, P1 and P2** — the generator comparison without the
  regional confound, 3 seeds each.
* **Track G, three arms, P3-ready** — the primary C-H1/C-H2 rows, 5 seeds. §18.3
  is explicit that the primary P3 Track-G generator hypotheses MUST use 5 seeds.
* **Track R, DET and LLM** — the secondary confirmatory rows, 3 seeds.
* **PromptHead ablation** — C-R-NOPROMPT, mandatory for C-H5, 3 seeds.

Two rules carry the scientific weight and are enforced rather than documented.
The seed family is fixed at 20260806-20260810 and a row may not use a seed
outside it — no "best seed" exists to be chosen. And a P3-ready row is *ready*,
not run: nothing in this module resolves a target label, a target metric or a
target path, and the P3 rows carry only the source-side configuration that C11
will later run predictions under.

The matrix identity covers the scientific rows alone. It does not move when a
backend, a machine or a clock changes, which is what lets a later full run on
the collaborator's GPU prove it executed this exact plan.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

SCHEMA_VERSION = "prism-c8-source-matrix-v1"

#: §18.3, the fixed seed family. Five for the primary P3 Track-G rows, the first
#: three for every 3-seed row. Extending this list after seeing a result is
#: exactly the cherry-picking the policy forbids.
SEED_FAMILY: tuple[int, ...] = (20260806, 20260807, 20260808, 20260809, 20260810)
SEEDS_5: tuple[int, ...] = SEED_FAMILY
SEEDS_3: tuple[int, ...] = SEED_FAMILY[:3]

#: §19: the three cross-domain protocols. P3's target is fixed to SiW-Mv2 v2 and
#: is named here only so a row can declare which protocol it belongs to; no path,
#: label or metric for it is reachable from this module.
P1, P2, P3 = "P1", "P2", "P3"
PROTOCOLS: dict[str, dict[str, Any]] = {
    P1: {"train": ("casia_fasd",), "dev": ("casia_fasd",), "test": ("msu_mfsd",),
         "role": "source-only cross-domain diagnostic"},
    P2: {"train": ("msu_mfsd",), "dev": ("msu_mfsd",), "test": ("casia_fasd",),
         "role": "source-only cross-domain diagnostic"},
    P3: {"train": ("casia_fasd", "msu_mfsd"), "dev": ("casia_fasd", "msu_mfsd"),
         "test": ("siw_mv2_v2",), "role": "fixed held-out target; predicted at C11 only"},
}

#: §18.1 arms. The treatment factor, and the only thing that varies across the
#: three primary Track-G rows.
ARMS: tuple[str, ...] = ("RND", "DET", "LLM")

#: §15.4 selection tuples, by protocol family.
P1P2_TUPLE: tuple[str, ...] = ("video_ACER", "video_BPCER", "NLL", "ECE", "epoch")
P3_READY_TUPLE: tuple[str, ...] = ("mean_domain_video_ACER", "max_domain_video_ACER",
                                   "mean_domain_video_BPCER", "mean_domain_NLL",
                                   "mean_domain_ECE", "epoch")

REPLICATION_ROLES: dict[str, dict[str, Any]] = {
    "hypothesis_critical": {"statistical_claim_allowed": True},
    "spec_mandated": {"statistical_claim_allowed": True},
    "diagnostic": {"statistical_claim_allowed": False},
}


class SourceMatrixError(ValueError):
    """The matrix cannot be planned as §18 declares it."""


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceRow:
    """One preregistered source-side run: method, protocol, arm, seed, config."""

    row_id: str
    experiment_id: str
    track: str
    arm: str
    protocol: str
    seed: int
    replication_role: str
    hypotheses: tuple[str, ...]
    flags: dict[str, Any]
    selection_tuple: tuple[str, ...]
    target_prediction_required: bool = False
    status: str = "PLANNED"
    blocked_reason: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.seed not in SEED_FAMILY:
            raise SourceMatrixError(
                f"row {self.row_id!r} uses seed {self.seed}, which is outside the fixed "
                f"family {SEED_FAMILY}. §18.3 forbids extending the family after a result "
                "is seen, so a seed outside it cannot be planned at all")
        if self.protocol not in PROTOCOLS:
            raise SourceMatrixError(f"unknown protocol {self.protocol!r}")
        if self.arm not in ARMS:
            raise SourceMatrixError(f"unknown arm {self.arm!r}")

    @property
    def config_identity(self) -> str:
        """Identity over the scientific configuration, excluding the seed.

        Rows that differ only by seed are replications of one configuration, and
        sharing a config identity is what makes "5 seeds of the same thing"
        checkable rather than asserted.
        """
        return _sha({"track": self.track, "arm": self.arm, "protocol": self.protocol,
                     "flags": self.flags, "selection_tuple": list(self.selection_tuple)})

    @property
    def run_identity(self) -> str:
        return _sha({"config_identity": self.config_identity, "seed": self.seed,
                     "experiment_id": self.experiment_id})

    def as_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "experiment_id": self.experiment_id,
                "track": self.track, "arm": self.arm, "protocol": self.protocol,
                "seed": self.seed, "replication_role": self.replication_role,
                "hypotheses": list(self.hypotheses), "flags": dict(self.flags),
                "selection_tuple": list(self.selection_tuple),
                "target_prediction_required": self.target_prediction_required,
                "status": self.status, "blocked_reason": self.blocked_reason,
                "config_identity": self.config_identity, "run_identity": self.run_identity,
                "statistical_claim_allowed": REPLICATION_ROLES.get(
                    self.replication_role, {}).get("statistical_claim_allowed", False),
                "notes": self.notes}


def _track_g_flags(arm: str) -> dict[str, Any]:
    from prism_fas.pipeline.adapters.c7 import TRACK_G_FLAGS

    return {**TRACK_G_FLAGS, "recipe_arm": arm}


def _track_r_flags(arm: str, *, prompt: str = "frozen_prompt") -> dict[str, Any]:
    from prism_fas.pipeline.adapters.c7 import TRACK_R_FLAGS

    return {**TRACK_R_FLAGS, "recipe_arm": arm, "prompt": prompt}


def plan_source_matrix() -> list[SourceRow]:
    """Every preregistered C8 row, in a deterministic order.

    Order is protocol, then track, then arm, then seed — stable so two plans
    built on two machines serialize identically and hash identically.
    """
    rows: list[SourceRow] = []

    # Track G, three arms, P1 and P2 — 3 seeds each (§18.1, §18.3).
    for protocol in (P1, P2):
        for arm in ARMS:
            for seed in SEEDS_3:
                rows.append(SourceRow(
                    row_id=f"C-G-{arm}-{protocol}-s{seed}",
                    experiment_id=f"C-G-{arm}", track="G", arm=arm, protocol=protocol,
                    seed=seed, replication_role="spec_mandated",
                    hypotheses=("C-H1", "C-H2"), flags=_track_g_flags(arm),
                    selection_tuple=P1P2_TUPLE,
                    notes="source-only cross-domain diagnostic; its test domain is "
                          "evaluation, never a tuning signal (§15.4)"))

    # Track G, three arms, P3-ready — 5 seeds (§18.3, mandatory for C-H1/C-H2).
    for arm in ARMS:
        for seed in SEEDS_5:
            rows.append(SourceRow(
                row_id=f"C-G-{arm}-P3READY-s{seed}",
                experiment_id=f"C-G-{arm}", track="G", arm=arm, protocol=P3, seed=seed,
                replication_role="hypothesis_critical", hypotheses=("C-H1", "C-H2"),
                flags=_track_g_flags(arm), selection_tuple=P3_READY_TUPLE,
                target_prediction_required=True,
                notes="P3-READY: trained and selected on CASIA-dev and MSU-dev only. No "
                      "SiW label, metric or path is resolvable at C8"))

    # Track R, DET and LLM — 3 seeds (§18.1, secondary confirmatory).
    for arm in ("DET", "LLM"):
        for seed in SEEDS_3:
            rows.append(SourceRow(
                row_id=f"C-R-{arm}-P3READY-s{seed}",
                experiment_id=f"C-R-{arm}", track="R", arm=arm, protocol=P3, seed=seed,
                replication_role="hypothesis_critical", hypotheses=("C-H3", "C-H4"),
                flags=_track_r_flags(arm), selection_tuple=P3_READY_TUPLE,
                target_prediction_required=True,
                notes="secondary confirmatory Track-R row"))

    # PromptHead ablation — C-R-NOPROMPT, 3 seeds, mandatory for C-H5 (§18.2).
    for seed in SEEDS_3:
        rows.append(SourceRow(
            row_id=f"C-R-NOPROMPT-P3READY-s{seed}",
            experiment_id="C-R-NOPROMPT", track="R", arm="LLM", protocol=P3, seed=seed,
            replication_role="hypothesis_critical", hypotheses=("C-H5",),
            flags=_track_r_flags("LLM", prompt="off"), selection_tuple=P3_READY_TUPLE,
            target_prediction_required=True,
            notes="C-H5: separates image augmentation from the semantic PromptHead effect. "
                  "lambda_P>0 versus lambda_P=0 under otherwise identical Track-R config"))
    return rows


def matrix_identity(rows: Sequence[SourceRow]) -> str:
    """Identity over the scientific rows alone — no clock, machine or backend."""
    return _sha([row.as_dict() for row in rows])


@dataclass
class SourceMatrixPlan:
    """The materialized plan plus the counts §18 requires it to satisfy."""

    rows: list[SourceRow] = field(default_factory=plan_source_matrix)

    @property
    def identity(self) -> str:
        return matrix_identity(self.rows)

    def by_experiment(self) -> dict[str, list[SourceRow]]:
        grouped: dict[str, list[SourceRow]] = {}
        for row in self.rows:
            grouped.setdefault(f"{row.experiment_id}:{row.protocol}", []).append(row)
        return grouped

    def seed_counts(self) -> dict[str, int]:
        return {key: len({row.seed for row in group})
                for key, group in self.by_experiment().items()}

    def validate(self) -> dict[str, Any]:
        """Check the plan against §18.3 rather than trusting the builder.

        Written as a checker rather than as assertions inside `plan_source_matrix`
        so a plan that violates the replication policy is *reportable* — a silent
        exception during planning would hide which row was wrong.
        """
        problems: list[str] = []
        counts = self.seed_counts()

        for arm in ARMS:
            key = f"C-G-{arm}:{P3}"
            if counts.get(key) != 5:
                problems.append(f"{key} has {counts.get(key)} seeds; §18.3 requires 5 for "
                                "the primary P3 Track-G generator hypotheses")
        for protocol in (P1, P2):
            for arm in ARMS:
                key = f"C-G-{arm}:{protocol}"
                if counts.get(key) != 3:
                    problems.append(f"{key} has {counts.get(key)} seeds; §18.3 requires 3")
        for experiment in ("C-R-DET", "C-R-LLM", "C-R-NOPROMPT"):
            key = f"{experiment}:{P3}"
            if counts.get(key) != 3:
                problems.append(f"{key} has {counts.get(key)} seeds; §18.3 requires 3")

        outside = [row.row_id for row in self.rows if row.seed not in SEED_FAMILY]
        if outside:
            problems.append(f"rows use seeds outside the fixed family: {outside}")

        single_seed_claims = [key for key, count in counts.items()
                              if count == 1 and any(
                                  row.replication_role in ("hypothesis_critical",
                                                           "spec_mandated")
                                  for row in self.by_experiment()[key])]
        if single_seed_claims:
            problems.append(f"single-seed rows carry a statistical claim: {single_seed_claims}")

        return {
            "schema_version": SCHEMA_VERSION,
            "matrix_identity": self.identity,
            "rows": len(self.rows),
            "unique_configurations": len({row.config_identity for row in self.rows}),
            "seed_counts": counts,
            "seed_family": list(SEED_FAMILY),
            "protocols": {name: dict(value) for name, value in PROTOCOLS.items()},
            "problems": problems,
            "valid": not problems,
            "replication_policy": (
                "primary P3 Track-G generator hypotheses use 5 seeds; P1/P2 Track-G "
                "comparisons and Track-R/PromptHead target comparisons use 3. Single-seed "
                "rows are diagnostic only and may not support superiority claims. No "
                "best-seed reporting (§18.3)"),
            "target_isolation": (
                "no row resolves a SiW label, metric or path. P3-ready means selected on "
                "CASIA-dev and MSU-dev only; prediction happens at C11 (§19.2)"),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.validate(), "rows_detail": [row.as_dict() for row in self.rows]}


def build_plan() -> SourceMatrixPlan:
    return SourceMatrixPlan()


__all__ = ["SCHEMA_VERSION", "SEED_FAMILY", "SEEDS_5", "SEEDS_3", "P1", "P2", "P3",
           "PROTOCOLS", "ARMS", "P1P2_TUPLE", "P3_READY_TUPLE", "REPLICATION_ROLES",
           "SourceMatrixError", "SourceRow", "plan_source_matrix", "matrix_identity",
           "SourceMatrixPlan", "build_plan"]
