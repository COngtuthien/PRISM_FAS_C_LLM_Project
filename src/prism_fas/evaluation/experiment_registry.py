"""The M10 experiment registry.

One durable record per matrix row: what was planned, what ran, on which backend,
which checkpoint was selected under the frozen source-side criterion, which
calibration froze it, which prediction lock it produced and how it ended.

Two rules the registry enforces rather than documents:

*   **A failed row is kept.** `COMPLETED`, `FAILED` and `BLOCKED` are the only
    terminal statuses, and a `FAILED` row keeps its failure record and still
    appears in the report. A row is never silently dropped.
*   **A row is launched once.** Asking to launch a row that already has a
    `RUNNING` or `COMPLETED` entry is refused here, not by hand.

The registry stores no target metric. Scoring results are bound to it by identity
only, after G8, so a registry read can never leak a target result into a selection
decision.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from prism_fas.utils.core import atomic_json_write, git_commit, utc_timestamp
from .contracts import (M10_REGISTRY_SCHEMA_VERSION, M10ContractError, STAGE_ORDER,
                        require_replication_role, require_status, stable_identity)

REGISTRY_FILE = "M10_REGISTRY.json"
STAGE_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED", "BLOCKED")
# The stages a TRAINING run can complete. G7 (target prediction) and G8 (scoring)
# are separate processes with separate permissions and are never part of a training
# run's lineage.
TRAINING_STAGES = ("G1", "G2", "G5", "G6")


@dataclass
class ExperimentRecord:
    """Everything M10 must be able to say about one row, later, from disk alone."""
    experiment_id: str
    category: str
    family: str
    variant: str
    seed: int | None
    replication_role: str
    status: str
    scientific_config_hash: str
    backend: str = "modal"
    run_id: str | None = None
    git_commit: str | None = None
    source_package_identity: str | None = None
    m8_bank_identity: str | None = None
    pretrained_identities: dict[str, str] = field(default_factory=dict)
    stage_statuses: dict[str, str] = field(default_factory=dict)
    best_checkpoint_path: str | None = None
    best_checkpoint_sha256: str | None = None
    source_dev_metrics: dict[str, Any] = field(default_factory=dict)
    source_calibration_sha256: str | None = None
    calibration_hash: str | None = None
    prediction_lock_identity: str | None = None
    scoring_identity: str | None = None
    failure: dict[str, Any] | None = None
    blocked_reason: str | None = None
    reference_run_id: str | None = None
    reuses_m9_reference: bool = False
    compute: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None

    def validate(self) -> "ExperimentRecord":
        require_status(self.status)
        require_replication_role(self.replication_role)
        for stage, value in self.stage_statuses.items():
            if stage not in STAGE_ORDER: raise M10ContractError(f"unknown stage {stage!r}")
            if value not in STAGE_STATUSES: raise M10ContractError(f"unknown stage status {value!r}")
        if self.status == "FAILED" and not self.failure:
            raise M10ContractError(f"{self.experiment_id}: a FAILED row must carry its failure record")
        if self.status == "BLOCKED" and not self.blocked_reason:
            raise M10ContractError(f"{self.experiment_id}: a BLOCKED row must carry its reason")
        if self.status == "COMPLETED" and not self.best_checkpoint_sha256 and not self.reuses_m9_reference:
            raise M10ContractError(f"{self.experiment_id}: a COMPLETED row must name its selected checkpoint")
        return self

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    """A JSON-backed registry with atomic writes and no in-place mutation of a
    terminal row."""

    def __init__(self, root: Path, records: dict[str, ExperimentRecord] | None = None,
                 matrix_identity: str = "") -> None:
        self.root = Path(root)
        self.records: dict[str, ExperimentRecord] = dict(records or {})
        self.matrix_identity = str(matrix_identity)

    # --- construction -------------------------------------------------------
    @classmethod
    def from_plan(cls, root: Path, plan: dict[str, Any]) -> "ExperimentRegistry":
        registry = cls(root, matrix_identity=str(plan["m10_matrix_identity"]))
        for row in plan["rows"]:
            registry.records[row["experiment_id"]] = ExperimentRecord(
                experiment_id=row["experiment_id"], category=row["category"], family=row["family"],
                variant=row["variant"], seed=row["seed"], replication_role=row["replication_role"],
                status=row["status"], scientific_config_hash=row["scientific_config_hash"],
                backend=row["backend"], blocked_reason=row.get("blocked_reason"),
                source_package_identity=row["source_package_identity"],
                m8_bank_identity=row["m8_bank_identity"],
                pretrained_identities={"siglip2": row["siglip2_identity"],
                                       "recipe_text_cache": row["recipe_text_cache_identity"],
                                       "convnext_weight": row["convnext_weight_sha256"],
                                       "m7_recipe_bank": row["m7_recipe_bank_identity"]},
                stage_statuses={stage: ("BLOCKED" if row["status"] == "BLOCKED" else "PENDING")
                                for stage in row["required_stages"]},
                reference_run_id=row.get("reference_run_id"),
                reuses_m9_reference=bool(row.get("reuses_m9_reference", False))).validate()
        return registry

    @classmethod
    def load(cls, root: Path) -> "ExperimentRegistry":
        import json
        path = Path(root) / REGISTRY_FILE
        if not path.is_file(): raise M10ContractError(f"{REGISTRY_FILE} does not exist under {root}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("registry_schema_version") != M10_REGISTRY_SCHEMA_VERSION:
            raise M10ContractError(f"registry schema {payload.get('registry_schema_version')!r} != "
                                   f"{M10_REGISTRY_SCHEMA_VERSION!r}")
        records = {str(row["experiment_id"]): ExperimentRecord(**row).validate()
                   for row in payload["records"]}
        return cls(Path(root), records, str(payload.get("m10_matrix_identity", "")))

    # --- mutation -----------------------------------------------------------
    def get(self, experiment_id: str) -> ExperimentRecord:
        if experiment_id not in self.records:
            raise M10ContractError(f"{experiment_id!r} is not in the registry")
        return self.records[experiment_id]

    def claim(self, experiment_id: str, *, run_id: str, backend: str) -> ExperimentRecord:
        """Mark a row RUNNING. Refuses a duplicate launch."""
        record = self.get(experiment_id)
        if record.status in ("RUNNING", "COMPLETED"):
            raise M10ContractError(f"{experiment_id} is already {record.status}; refusing a duplicate launch")
        if record.status == "BLOCKED":
            raise M10ContractError(f"{experiment_id} is BLOCKED: {record.blocked_reason}")
        record.status, record.run_id, record.backend = "RUNNING", str(run_id), str(backend)
        record.git_commit = git_commit(Path.cwd())
        record.updated_at = utc_timestamp()
        return record.validate()

    def update(self, experiment_id: str, **fields: Any) -> ExperimentRecord:
        record = self.get(experiment_id)
        if "status" in fields: require_status(str(fields["status"]))
        for key, value in fields.items():
            if not hasattr(record, key): raise M10ContractError(f"unknown registry field {key!r}")
            setattr(record, key, value)
        record.updated_at = utc_timestamp()
        return record.validate()

    def set_stage(self, experiment_id: str, stage: str, status: str) -> ExperimentRecord:
        if stage not in STAGE_ORDER: raise M10ContractError(f"unknown stage {stage!r}")
        if status not in STAGE_STATUSES: raise M10ContractError(f"unknown stage status {status!r}")
        record = self.get(experiment_id)
        record.stage_statuses[stage] = status
        record.updated_at = utc_timestamp()
        return record.validate()

    def fail(self, experiment_id: str, *, stage: str, error: str, recoverable: bool = False) -> ExperimentRecord:
        """A failure is recorded, never dropped."""
        record = self.get(experiment_id)
        record.status = "FAILED"
        record.failure = {"stage": stage, "error": str(error), "recoverable": bool(recoverable),
                          "recorded_at": utc_timestamp()}
        if stage in record.stage_statuses: record.stage_statuses[stage] = "FAILED"
        record.updated_at = utc_timestamp()
        return record.validate()

    # --- views --------------------------------------------------------------
    def by_status(self, status: str) -> list[ExperimentRecord]:
        return [record for record in self.ordered() if record.status == require_status(status)]

    def ordered(self) -> list[ExperimentRecord]:
        return [self.records[key] for key in sorted(self.records)]

    def completed_predictions(self) -> list[ExperimentRecord]:
        return [record for record in self.ordered()
                if record.status == "COMPLETED" and record.prediction_lock_identity]

    def identity(self) -> str:
        """Content identity of the registry's scientific fields.

        Excludes wall-clock, run ids, git commit, backend and compute so two
        registries describing the same science compare equal.
        """
        volatile = {"updated_at", "run_id", "git_commit", "compute", "backend",
                    "best_checkpoint_path", "reference_run_id"}
        return stable_identity({"schema_version": M10_REGISTRY_SCHEMA_VERSION,
                                "records": [{key: value for key, value in record.payload().items()
                                             if key not in volatile} for record in self.ordered()]})

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.ordered(): counts[record.status] = counts.get(record.status, 0) + 1
        return {"registry_schema_version": M10_REGISTRY_SCHEMA_VERSION,
                "m10_matrix_identity": self.matrix_identity, "registry_identity": self.identity(),
                "records": len(self.records), "by_status": dict(sorted(counts.items())),
                "failed": [record.experiment_id for record in self.by_status("FAILED")],
                "blocked": [record.experiment_id for record in self.by_status("BLOCKED")],
                "with_prediction_lock": len(self.completed_predictions())}

    # --- persistence --------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        return {"registry_schema_version": M10_REGISTRY_SCHEMA_VERSION,
                "m10_matrix_identity": self.matrix_identity,
                "registry_identity": self.identity(),
                "records": [record.payload() for record in self.ordered()]}

    def write(self, *, dry_run: bool = False) -> Path:
        path = self.root / REGISTRY_FILE
        if not dry_run: atomic_json_write(path, self.payload())
        return path


def validate_m9_reference_binding(row: dict[str, Any], run_summary: dict[str, Any],
                                  expected: dict[str, str]) -> dict[str, Any]:
    """Decide, from the RECORDED evidence, whether `B08-s20260806` may bind the M9
    reference run instead of retraining it.

    Experiment contract §2.1 calls this an identity binding, not an approximation:
    the row is accepted only if every recorded scientific identity and the seed
    match. If any of them differs the binding is REFUSED and the row must be
    retrained under a new run id — this function never reports a near-match as a
    match, and it reads the M9 run's own `run.json` rather than recomputing what it
    would have been.
    """
    identity = dict(run_summary.get("identity") or {})
    checks: dict[str, dict[str, Any]] = {}
    for name, value in sorted(expected.items()):
        checks[name] = {"m9_reference": identity.get(name), "m10_row": value,
                        "matches": identity.get(name) == value}
    lineage = [(str(entry["stage"]), str(entry.get("status"))) for entry in run_summary.get("stage_lineage") or []]
    completed = [stage for stage, status in lineage if status == "COMPLETED"]
    # Only the TRAINING stages can be satisfied by a training run. G7 is the M10
    # target-prediction stage: it is still pending for every row in the matrix, and
    # requiring it here would refuse a binding for the one reason that is true of
    # every row alike.
    training_stages = [stage for stage in (row.get("required_stages") or []) if stage in TRAINING_STAGES]
    structural = {
        "seed_matches": {"m9_reference": run_summary.get("seed"), "m10_row": row.get("seed"),
                         "matches": int(run_summary.get("seed", -1)) == int(row.get("seed", -2))},
        "run_completed": {"m9_reference": run_summary.get("status"), "m10_row": "COMPLETED",
                          "matches": run_summary.get("status") == "COMPLETED"},
        "every_declared_training_stage_completed": {
            "m9_reference": completed, "m10_row": training_stages,
            "matches": bool(training_stages) and all(stage in completed for stage in training_stages)},
        "target_prediction_still_pending": {
            "m9_reference": [stage for stage, _ in lineage], "m10_row": "G7 has not run for any row",
            "matches": "G7" not in [stage for stage, _ in lineage]},
        "target_never_opened": {"m9_reference": run_summary.get("target_test_opened"),
                                "m10_row": False,
                                "matches": run_summary.get("target_test_opened") is False}}
    checks.update(structural)
    mismatched = sorted(name for name, check in checks.items() if not check["matches"])
    return {"binding_schema_version": "m10-m9-reference-binding-v1",
            "experiment_id": row["experiment_id"],
            "reference_run_id": row.get("reference_run_id"),
            "checks": checks, "mismatched": mismatched,
            "binding_accepted": not mismatched,
            "decision": ("bind the existing M9 reference run" if not mismatched else
                         "REFUSED: retrain under a new run id and document why"),
            "target_labels_opened": False}


def source_matrix_lock(registry: ExperimentRegistry, plan: dict[str, Any]) -> dict[str, Any]:
    """Freeze every selected checkpoint and calibration BEFORE the first G7 run.

    After this lock exists, no source-side selection may change; a target result
    therefore cannot reach back into a selection decision even in principle.
    """
    entries = []
    for record in registry.ordered():
        if record.status != "COMPLETED": continue
        if not (record.best_checkpoint_sha256 and record.source_calibration_sha256):
            raise M10ContractError(f"{record.experiment_id} is COMPLETED but has no frozen "
                                   f"checkpoint/calibration to lock")
        entries.append({"experiment_id": record.experiment_id, "seed": record.seed,
                        "scientific_config_hash": record.scientific_config_hash,
                        "best_checkpoint_sha256": record.best_checkpoint_sha256,
                        "source_calibration_sha256": record.source_calibration_sha256,
                        "calibration_hash": record.calibration_hash,
                        "source_dev_metrics": record.source_dev_metrics})
    body = {"source_matrix_lock_schema_version": "m10-source-matrix-lock-v1",
            "m10_matrix_identity": plan["m10_matrix_identity"],
            "registry_identity": registry.identity(), "entries": entries,
            "entry_count": len(entries), "target_labels_opened": False,
            "selection_used_target": False}
    return {**body, "source_matrix_lock_identity": stable_identity(body)}
