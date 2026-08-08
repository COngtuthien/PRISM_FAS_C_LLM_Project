"""The deterministic M10 experiment matrix planner.

Materializes `configs/experiments/m10_matrix.yaml` — the Table 59 baselines and the
Table 60 ablations — into an ordered list of logical rows plus one
`m10_matrix_identity`.

Two properties this module exists to guarantee:

1.  **Determinism.** Planning twice produces identical rows and an identical
    identity. The identity is derived from the canonical SCIENTIFIC fields only:
    no timestamp, no machine path, no run directory, no Modal id, no host.
2.  **Nothing silently disappears.** A blocked ablation stays in the matrix as a
    `BLOCKED` row carrying its reason. A baseline is never re-pointed at a
    neighbouring configuration because a flag combination is inconvenient.

Every baseline is a CONFIGURATION of the shared implementation, expressed as a
delta against the frozen B08 reference flag set — never a forked model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
from .contracts import (M10_MATRIX_SCHEMA_VERSION, M10ContractError, require_replication_role,
                        require_status, stable_identity)

# Flag consistency rules. A row that breaks one of these is a contradiction in the
# declared matrix, not something to repair silently at plan time.
FLAG_KEYS = ("local_branch", "global_branch", "fusion", "region", "manifold", "prototype_k",
             "synthetic", "recipe_conditioning", "quality_weighting", "outlier_loss",
             "prompt", "sampler", "frames_per_video")


@dataclass(frozen=True)
class ExperimentRow:
    """One logical scientific row of the matrix."""
    experiment_id: str
    category: str                      # baseline | ablation
    family: str
    variant: str
    seed: int | None
    flags: dict[str, Any]
    replication_role: str
    status: str
    required_stages: tuple[str, ...]
    parent_experiment_id: str | None
    source_selection_rule: dict[str, Any]
    target_prediction_required: bool
    frozen_inputs: dict[str, str]
    scientific_config_hash: str
    hypotheses: tuple[str, ...] = ()
    backend: str = "modal"
    protocol: str | None = None
    blocked_reason: str | None = None
    reference_run_id: str | None = None
    reuses_m9_reference: bool = False
    table_59_text: str | None = None
    question: str | None = None

    def canonical(self) -> dict[str, Any]:
        """Exactly the fields `m10_matrix_identity` is allowed to see.

        No timestamp, no path, no run directory, no Modal id, no host, and no
        execution-protocol attribute.
        """
        return {"experiment_id": self.experiment_id, "category": self.category,
                "family": self.family, "variant": self.variant, "seed": self.seed,
                "scientific_config_hash": self.scientific_config_hash,
                "parent_experiment_id": self.parent_experiment_id,
                "source_package_identity": self.frozen_inputs["source_package_identity"],
                "m8_bank_identity": self.frozen_inputs["m8_bank_identity"],
                "m7_recipe_bank_identity": self.frozen_inputs["m7_recipe_bank_identity"],
                "siglip2_identity": self.frozen_inputs["siglip2_identity"],
                "recipe_text_cache_identity": self.frozen_inputs["recipe_text_cache_identity"],
                "convnext_weight_sha256": self.frozen_inputs["convnext_weight_sha256"],
                "required_stages": list(self.required_stages),
                "source_selection_rule": self.source_selection_rule,
                "target_prediction_required": self.target_prediction_required,
                "replication_role": self.replication_role, "status": self.status,
                "blocked_reason": self.blocked_reason, "hypotheses": list(self.hypotheses),
                "backend": self.backend}

    def payload(self) -> dict[str, Any]:
        return {**self.canonical(), "flags": dict(self.flags), "protocol": self.protocol,
                "reference_run_id": self.reference_run_id,
                "reuses_m9_reference": self.reuses_m9_reference,
                "table_59_text": self.table_59_text, "question": self.question}


def _check_flags(experiment_id: str, flags: dict[str, Any], vocabulary: dict[str, list[Any]]) -> None:
    unknown = sorted(set(flags) - set(FLAG_KEYS))
    if unknown: raise M10ContractError(f"{experiment_id}: unknown flags {unknown}")
    missing = sorted(set(FLAG_KEYS) - set(flags))
    if missing: raise M10ContractError(f"{experiment_id}: unresolved flags {missing}")
    for key, value in flags.items():
        allowed = vocabulary.get(key)
        if allowed is not None and value not in allowed:
            raise M10ContractError(f"{experiment_id}: flag {key}={value!r} is outside {allowed}")
    region, manifold, k = flags["region"], flags["manifold"], int(flags["prototype_k"])
    fusion, prompt, outlier = flags["fusion"], flags["prompt"], flags["outlier_loss"]
    synthetic, quality, recipe = flags["synthetic"], flags["quality_weighting"], flags["recipe_conditioning"]
    branches = (flags["local_branch"] != "off") + (flags["global_branch"] != "off")
    problems = []
    if (manifold == "off") != (k == 0): problems.append("manifold=off must pair with prototype_k=0")
    if manifold == "global_center" and k != 1: problems.append("a global center is exactly one prototype")
    if manifold == "multi_prototype" and (k < 1 or region != "on"):
        problems.append("multi-prototype manifolds are regional and need at least one prototype")
    if region == "on" and fusion != "prism_noisy_or":
        problems.append("the regional detector uses the Table 34 fusion")
    if region == "off" and fusion == "prism_noisy_or":
        problems.append("the Table 34 fusion needs the regional evidence term")
    if prompt != "off" and (region != "on" or flags["global_branch"] == "off"):
        problems.append("PromptHead is defined over region embeddings and the frozen text tower")
    if outlier == "mask_aware" and region != "on":
        problems.append("a mask-aware outlier loss needs region masks")
    if synthetic == "none" and quality != "off":
        problems.append("q weights synthetic samples; with no synthetic data there is nothing to weight")
    if quality != "off" and synthetic == "none":
        problems.append("quality weighting requires synthetic samples")
    if recipe != "off" and synthetic == "none":
        problems.append("recipe conditioning requires synthetic samples")
    if fusion == "single_logit" and branches != 1:
        problems.append("a single-logit baseline has exactly one backbone branch")
    if fusion in ("simple_concat", "prism_noisy_or") and branches != 2:
        problems.append(f"{fusion} needs both backbone branches")
    if problems: raise M10ContractError(f"{experiment_id}: contradictory flags: {problems}")


def _stages(config: dict[str, Any], flags: dict[str, Any], *, role: str, status: str) -> tuple[str, ...]:
    stages = config["stages"]
    if status == "BLOCKED": return tuple(stages["blocked"])
    if role == "parity": return tuple(stages["parity"])
    return tuple(stages["with_manifold"] if flags["manifold"] != "off" else stages["base"])


def _seeds(config: dict[str, Any], role: str) -> tuple[int | None, ...]:
    policy = config["replication_roles"][role]
    count = int(policy["seeds"])
    if count == 0: return (None,)
    canonical = [int(value) for value in config["seeds"]["canonical"]]
    if count > len(canonical):
        raise M10ContractError(f"role {role!r} asks for {count} seeds but only {len(canonical)} are declared")
    if count == 1: return (int(config["seeds"]["single_seed_default"]),)
    return tuple(canonical[:count])


def _scientific_hash(flags: dict[str, Any], seed: int | None, config: dict[str, Any]) -> str:
    """The science only: the resolved flag set, the seed, the frozen inputs and the
    source-side selection rule.

    Excludes `backend` and every execution-protocol attribute (`protocol`,
    `parity_steps`, `required_stages`, `status`), so the A09 parity row hashes
    identically to B08 seed 20260806 — which is the definition of the parity test.
    `required_stages` is excluded because it is DERIVED from the flags and the
    replication role, so including it would let an execution attribute change a
    scientific identity.
    """
    return stable_identity({"schema_version": M10_MATRIX_SCHEMA_VERSION,
                            "flags": {key: flags[key] for key in FLAG_KEYS}, "seed": seed,
                            "frozen_inputs": dict(config["frozen_inputs"]),
                            "source_selection_rule": dict(config["source_selection_rule"])})


def load_matrix_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise M10ContractError(f"{Path(path).name} is not a mapping")
    if payload.get("matrix_schema_version") != M10_MATRIX_SCHEMA_VERSION:
        raise M10ContractError(f"matrix schema {payload.get('matrix_schema_version')!r} != "
                               f"{M10_MATRIX_SCHEMA_VERSION!r}")
    for name in ("seeds", "replication_roles", "frozen_inputs", "source_selection_rule",
                 "flag_vocabulary", "reference_flags", "baselines", "ablations", "hypotheses", "stages"):
        if name not in payload: raise M10ContractError(f"the matrix config is missing {name!r}")
    return payload


def _rows_for(entry: dict[str, Any], *, category: str, family: str, variant: str,
              config: dict[str, Any], parent: str | None, prefix: str) -> list[ExperimentRow]:
    reference = dict(config["reference_flags"])
    flags = {**reference, **{str(key): value for key, value in (entry.get("flags") or {}).items()}}
    role = require_replication_role(str(entry.get("replication_role", "diagnostic")))
    status = require_status(str(entry.get("status", "BLOCKED" if role == "blocked" else "PLANNED")))
    if (role == "blocked") != (status == "BLOCKED"):
        raise M10ContractError(f"{prefix}: replication role {role!r} and status {status!r} disagree")
    blocked_reason = entry.get("blocked_reason")
    if status == "BLOCKED" and not blocked_reason:
        raise M10ContractError(f"{prefix}: a BLOCKED row must carry its reason")
    if status != "BLOCKED" and blocked_reason:
        raise M10ContractError(f"{prefix}: only a BLOCKED row may carry a blocked reason")
    _check_flags(prefix, flags, config["flag_vocabulary"])
    stages = _stages(config, flags, role=role, status=status)
    frozen = {str(key): str(value) for key, value in config["frozen_inputs"].items()}
    rows: list[ExperimentRow] = []
    for seed in _seeds(config, role):
        experiment_id = prefix if seed is None else f"{prefix}-s{seed}"
        reuse = (entry.get("reference_run_id") is not None
                 and seed == int(config["seeds"]["single_seed_default"]))
        rows.append(ExperimentRow(
            experiment_id=experiment_id, category=category, family=family, variant=variant, seed=seed,
            flags=flags, replication_role=role, status=status, required_stages=stages,
            parent_experiment_id=parent,
            source_selection_rule=dict(config["source_selection_rule"]),
            target_prediction_required=(status != "BLOCKED" and role != "parity"),
            frozen_inputs=frozen,
            scientific_config_hash=_scientific_hash(flags, seed, config),
            hypotheses=tuple(str(name) for name in (entry.get("hypotheses") or ())),
            backend=str(entry.get("backend", "modal")),
            protocol=entry.get("protocol"), blocked_reason=blocked_reason,
            reference_run_id=(str(entry["reference_run_id"]) if reuse else None),
            reuses_m9_reference=bool(reuse),
            table_59_text=entry.get("table_59_text"), question=entry.get("question")))
    return rows


def plan_matrix(config: dict[str, Any]) -> list[ExperimentRow]:
    """Materialize every logical row, in a deterministic order."""
    rows: list[ExperimentRow] = []
    for entry in config["baselines"]:
        identifier = str(entry["id"])
        rows += _rows_for(entry, category="baseline", family=identifier, variant="reference",
                          config=config, parent=entry.get("parent"), prefix=identifier)
    for block in config["ablations"]:
        identifier, family = str(block["id"]), str(block["family"])
        for variant in block["variants"]:
            name = str(variant["variant"])
            rows += _rows_for({**variant, "question": block.get("question")}, category="ablation",
                              family=family, variant=name, config=config,
                              parent=str(block["parent"]), prefix=f"{identifier}-{family}-{name}")
    identifiers = [row.experiment_id for row in rows]
    duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if duplicates: raise M10ContractError(f"duplicate experiment ids: {duplicates}")
    return rows


def matrix_identity(rows: list[ExperimentRow]) -> str:
    return stable_identity({"schema_version": M10_MATRIX_SCHEMA_VERSION,
                            "rows": [row.canonical() for row in rows]})


def summarize(rows: list[ExperimentRow], config: dict[str, Any]) -> dict[str, Any]:
    """The counts a reader needs before anything expensive is launched."""
    executable = [row for row in rows if row.status != "BLOCKED"]
    training = [row for row in executable if row.replication_role != "parity"]
    by_role: dict[str, int] = {}
    for row in rows: by_role[row.replication_role] = by_role.get(row.replication_role, 0) + 1
    by_status: dict[str, int] = {}
    for row in rows: by_status[row.status] = by_status.get(row.status, 0) + 1
    blocked = [{"experiment_id": row.experiment_id, "reason": row.blocked_reason}
               for row in rows if row.status == "BLOCKED"]
    return {"logical_rows": len(rows), "executable_rows": len(executable),
            "planned_training_runs": len(training),
            # B08 seed 20260806 binds the existing M9 reference run instead of
            # retraining it, so it is planned but costs no new GPU work.
            "planned_new_training_runs": sum(1 for row in training if not row.reuses_m9_reference),
            "planned_parity_runs": len(executable) - len(training),
            "blocked_rows": len(blocked), "blocked": blocked,
            "rows_by_replication_role": dict(sorted(by_role.items())),
            "rows_by_status": dict(sorted(by_status.items())),
            "unique_scientific_configs": len({row.scientific_config_hash for row in rows}),
            "target_predictions_required": sum(1 for row in rows if row.target_prediction_required),
            "seeds": list(config["seeds"]["canonical"]),
            "claim_bearing_rows": sum(1 for row in rows
                                      if row.replication_role in ("spec_mandated", "hypothesis_critical")),
            "reuses_m9_reference": [row.experiment_id for row in rows if row.reuses_m9_reference]}


def validate_hypotheses(rows: list[ExperimentRow], config: dict[str, Any]) -> dict[str, Any]:
    """Every declared comparison must name rows that exist and can carry a claim."""
    families = {row.family for row in rows} | {row.experiment_id for row in rows}
    prefixes = {row.experiment_id.rsplit("-s", 1)[0] for row in rows}
    problems, checked = [], {}
    for name, block in config["hypotheses"].items():
        for side in ("treatment", "control"):
            reference = str(block[side])
            if reference not in families and reference not in prefixes:
                problems.append(f"{name}.{side} names {reference!r}, which is not in the matrix")
                continue
            matching = [row for row in rows
                        if row.family == reference or row.experiment_id.rsplit("-s", 1)[0] == reference]
            if block.get("kind") != "parity_not_superiority" and not all(
                    row.replication_role in ("spec_mandated", "hypothesis_critical") for row in matching):
                problems.append(f"{name}.{side} ({reference}) is not a claim-bearing row")
        checked[name] = {"treatment": block["treatment"], "control": block["control"],
                         "kind": block.get("kind", "superiority")}
    if problems: raise M10ContractError(f"the declared hypothesis family is inconsistent: {problems}")
    return checked


def build_plan(config_path: Path) -> dict[str, Any]:
    """The complete plan payload written to `reports/m10/M10_MATRIX_PLAN.json`."""
    config = load_matrix_config(Path(config_path))
    rows = plan_matrix(config)
    return {"matrix_schema_version": M10_MATRIX_SCHEMA_VERSION,
            "m10_matrix_identity": matrix_identity(rows),
            "config_identity_sha256": stable_identity(config),
            "frozen_inputs": {str(k): str(v) for k, v in config["frozen_inputs"].items()},
            "replication_policy": {"seeds": dict(config["seeds"]),
                                   "roles": dict(config["replication_roles"])},
            "hypotheses": validate_hypotheses(rows, config),
            "summary": summarize(rows, config),
            "rows": [row.payload() for row in rows]}


def plans_agree(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare two independent planning passes. Determinism is proved, not assumed."""
    left = [row["experiment_id"] for row in first["rows"]]
    right = [row["experiment_id"] for row in second["rows"]]
    mismatched = [row_a["experiment_id"] for row_a, row_b in zip(first["rows"], second["rows"])
                  if row_a != row_b]
    identical = (first["m10_matrix_identity"] == second["m10_matrix_identity"]
                 and left == right and not mismatched)
    return {"identical": bool(identical),
            "identity_matches": first["m10_matrix_identity"] == second["m10_matrix_identity"],
            "row_order_matches": left == right, "mismatched_rows": mismatched,
            "m10_matrix_identity": first["m10_matrix_identity"]}
