"""M10 additive SiW-Mv2 target evaluation package.

This module builds a SEPARATE, versioned, immutable target-only artifact pair. It
never touches `prism_data_v1_m3b`, `configs/data/siw_mv2.yaml`, the M8 bank or any
M9 artifact, so every identity those bind stays exactly as it is.

    FEATURE package   label-free, usable by G7
    LABEL artifact    evaluator-only, usable only by G8, physically separate

Two rules this module exists to enforce:

*   **Nothing is inferred from a path at scoring time.** The dataset's own
    directory organization is read exactly ONCE, here, by the evaluator-only path,
    and is converted immediately into an opaque-id-keyed table. Every attack family
    and every filename stem is DECLARED in `configs/data/siw_mv2_target_v2.yaml`
    and checked; an unlisted family or an unexpected stem is a hard failure, never
    a silently new class. A plausible-looking recursive glob previously dropped
    596 of 915 spoof videos without a word.
*   **The frozen live population must reproduce byte-for-byte.** `adapter_version`
    stays `"1.0"` because `preprocess_m2.sample_id` hashes it; the layout revision
    is carried by `layout_rules_version` and the new package identity instead. If
    the same frozen live frame produces a different `crop_sha256` under unchanged
    preprocessing, that is a defect to investigate, never a threshold to relax.
"""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import yaml
from pydantic import BaseModel, ConfigDict
from prism_fas.config.models import DatasetDefinition
from prism_fas.data.adapters.adapters import opaque_record_id
from prism_fas.utils.core import atomic_json_write, stable_json_hash

TARGET_EVAL_SCHEMA_VERSION = "m10-target-eval-v1"
LABEL_SCHEMA_VERSION = "m10-target-labels-v1"
TARGET_PROFILE = "target_eval_v2"
VIDEO_SUFFIXES = (".avi", ".mov", ".mp4")

# A feature-side row may never carry any of these.
FORBIDDEN_FEATURE_FIELDS = frozenset({
    "label", "label_live_spoof", "true_label", "target", "class_target",
    "attack_type", "attack_family", "taxonomy", "private_label", "subject_id",
    "official_split", "identity_embedding", "source_path"})


class TargetEvalError(ValueError):
    """The target evaluation layout, inventory or package is not as declared."""


class TargetLayoutV2(BaseModel):
    """`configs/data/siw_mv2_target_v2.yaml`, strictly.

    `DatasetDefinition` forbids extra keys, so the v2 layout — which adds the
    declared family/stem map, the expected counts and the layout revision — gets
    its own strict model and projects down to a `DatasetDefinition` for the
    unchanged adapter.
    """
    model_config = ConfigDict(extra="forbid")
    dataset: str
    adapter_version: str
    layout_rules_version: str
    root_relative: str
    include_globs: list[str]
    path_pattern: str
    official_split: str
    private_evaluation_fields: list[str]
    attack_family_stems: dict[str, str]
    expected_counts: dict[str, Any]
    blocked_reason: str | None = None

    def validate_contract(self) -> "TargetLayoutV2":
        if self.dataset != "siw_mv2": raise TargetEvalError("the target layout is siw_mv2 only")
        # FROZEN. `preprocess_m2.sample_id` hashes the adapter version, so bumping
        # it would change every one of the 785 frozen live sample ids and forfeit
        # the byte-reproduction gate that is this package's primary acceptance test.
        if self.adapter_version != "1.0":
            raise TargetEvalError(f"adapter_version must stay '1.0', got {self.adapter_version!r}; "
                                  f"the layout revision is carried by layout_rules_version")
        if self.layout_rules_version == "siw-mv2-layout-v1":
            raise TargetEvalError("the v2 layout must declare its own layout_rules_version")
        declared = self.expected_counts.get("by_attack_family") or {}
        if set(declared) != set(self.attack_family_stems):
            raise TargetEvalError("declared family counts and family stems disagree")
        if sum(declared.values()) != int(self.expected_counts["spoof"]):
            raise TargetEvalError("declared per-family counts do not sum to the declared spoof total")
        if int(self.expected_counts["live"]) + int(self.expected_counts["spoof"]) != int(self.expected_counts["total"]):
            raise TargetEvalError("declared live + spoof does not equal the declared total")
        return self

    def dataset_definition(self) -> DatasetDefinition:
        return DatasetDefinition.model_validate({
            "dataset": self.dataset, "adapter_version": self.adapter_version,
            "root_relative": self.root_relative, "include_globs": list(self.include_globs),
            "path_pattern": self.path_pattern, "official_split": self.official_split,
            "private_evaluation_fields": list(self.private_evaluation_fields),
            "blocked_reason": self.blocked_reason})

    def layout_identity(self) -> str:
        """Content identity of the LAYOUT RULES. No path, no machine, no clock."""
        return stable_json_hash({"schema_version": TARGET_EVAL_SCHEMA_VERSION,
                                 **self.model_dump(mode="json")})


def load_target_layout(path: Path) -> TargetLayoutV2:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise TargetEvalError(f"{Path(path).name} is not a mapping")
    return TargetLayoutV2.model_validate(payload).validate_contract()


# --- inventory ---------------------------------------------------------------

@dataclass(frozen=True)
class InventoryEntry:
    """One declared target video. `attack_family` is EVALUATOR-ONLY."""
    video_id: str                 # opaque siw_<16 hex>
    relative_path: str            # dataset-root-relative, evaluator-only
    label: str                    # live | spoof, evaluator-only
    attack_family: str | None     # evaluator-only
    stem: str


@dataclass
class InventoryAudit:
    entries: list[InventoryEntry] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    undeclared_family: list[str] = field(default_factory=list)
    stem_mismatch: list[dict[str, str]] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    count_mismatch: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not (self.unmatched or self.undeclared_family or self.stem_mismatch
                    or self.duplicate_ids or self.count_mismatch)

    def by_family(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.attack_family: counts[entry.attack_family] = counts.get(entry.attack_family, 0) + 1
        return dict(sorted(counts.items()))

    def counts(self) -> dict[str, int]:
        return {"total": len(self.entries),
                "live": sum(1 for entry in self.entries if entry.label == "live"),
                "spoof": sum(1 for entry in self.entries if entry.label == "spoof"),
                "families": len(self.by_family())}

    def report(self) -> dict[str, Any]:
        return {"schema_version": TARGET_EVAL_SCHEMA_VERSION, "passed": self.passed,
                "counts": self.counts(), "by_attack_family": self.by_family(),
                "unmatched": self.unmatched[:20], "unmatched_count": len(self.unmatched),
                "undeclared_family": sorted(set(self.undeclared_family)),
                "stem_mismatch": self.stem_mismatch[:20], "stem_mismatch_count": len(self.stem_mismatch),
                "duplicate_ids": self.duplicate_ids[:20], "duplicate_id_count": len(self.duplicate_ids),
                "count_mismatch": self.count_mismatch,
                "inventory_identity_sha256": self.identity()}

    def identity(self) -> str:
        """Identity of the LOGICAL inventory: opaque ids only, sorted.

        Deliberately excludes the relative path, the label and the family, so the
        feature side can bind an inventory identity without that binding requiring
        a label.
        """
        return stable_json_hash({"schema_version": TARGET_EVAL_SCHEMA_VERSION,
                                 "video_ids": sorted(entry.video_id for entry in self.entries)})


def discover_videos(raw_root: Path, layout: TargetLayoutV2) -> tuple[Path, list[Path]]:
    base = Path(raw_root) / layout.root_relative
    if not base.is_dir(): raise TargetEvalError(f"the SiW-Mv2 layout root does not exist: {base}")
    found: set[Path] = set()
    for pattern in layout.include_globs:
        found.update(path for path in base.glob(pattern) if path.is_file())
    return base, sorted(found, key=lambda path: path.as_posix())


def all_video_files(raw_root: Path, layout: TargetLayoutV2) -> list[Path]:
    """Every video under the layout root, whatever the declared globs match.

    Used to prove the declared globs miss nothing: the audit compares this against
    what the rules matched, which is exactly the check the v1 layout never had.
    """
    base = Path(raw_root) / layout.root_relative
    return sorted((path for path in base.rglob("*")
                   if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
                  key=lambda path: path.as_posix())


def audit_inventory(raw_root: Path, layout: TargetLayoutV2) -> InventoryAudit:
    """Match every declared video, and fail on anything undeclared."""
    base, matched = discover_videos(raw_root, layout)
    pattern = re.compile(layout.path_pattern)
    audit = InventoryAudit()
    seen: dict[str, str] = {}
    for path in matched:
        relative = path.relative_to(base).as_posix()
        match = pattern.fullmatch(relative)
        if match is None:
            audit.unmatched.append(relative); continue
        groups = match.groupdict()
        if groups.get("live_label"):
            label, family, stem = "live", None, "Live"
        else:
            label, family, stem = "spoof", groups.get("attack_family"), groups.get("stem")
            declared = layout.attack_family_stems.get(family or "")
            if declared is None:
                audit.undeclared_family.append(family or "<none>"); continue
            if stem != declared:
                audit.stem_mismatch.append({"family": family or "", "declared_stem": declared,
                                            "actual_stem": stem or ""})
                continue
        video_id = opaque_record_id("siw_mv2", relative)
        if seen.setdefault(video_id, relative) != relative:
            audit.duplicate_ids.append(video_id); continue
        audit.entries.append(InventoryEntry(video_id=video_id, relative_path=relative, label=label,
                                            attack_family=family, stem=stem or ""))
    # Anything on disk the declared rules did not match is an unmatched video, not
    # an absence. This is what the v1 layout failed to notice for 915 videos.
    every = {path.relative_to(base).as_posix() for path in all_video_files(raw_root, layout)}
    audit.unmatched.extend(sorted(every - {path.relative_to(base).as_posix() for path in matched}))
    expected = layout.expected_counts
    observed = audit.counts()
    declared_families = {str(k): int(v) for k, v in (expected.get("by_attack_family") or {}).items()}
    mismatch: dict[str, Any] = {}
    for key in ("total", "live", "spoof"):
        if observed[key] != int(expected[key]): mismatch[key] = {"expected": int(expected[key]), "actual": observed[key]}
    families = audit.by_family()
    differing = {name: {"expected": count, "actual": families.get(name, 0)}
                 for name, count in declared_families.items() if families.get(name, 0) != count}
    if differing: mismatch["by_attack_family"] = differing
    audit.count_mismatch = mismatch
    return audit


# --- evaluation-only label artifact -------------------------------------------

def evaluation_label_rows(audit: InventoryAudit) -> list[dict[str, Any]]:
    """Canonicalize labels ONCE, here, keyed by the opaque id and nothing else.

    G8 never infers a label from a filename, a path or a directory: it reads this
    table. The relative path is deliberately NOT carried into it.
    """
    if not audit.passed: raise TargetEvalError("refusing to build labels from a failed inventory audit")
    return sorted(({"video_id": entry.video_id, "label": entry.label,
                    "attack_family": entry.attack_family or "live"} for entry in audit.entries),
                  key=lambda row: row["video_id"])


def label_artifact_identity(rows: Sequence[dict[str, Any]], *, layout_identity: str) -> str:
    return stable_json_hash({"schema_version": LABEL_SCHEMA_VERSION, "layout_identity": layout_identity,
                             "rows": [[row["video_id"], row["label"], row["attack_family"]]
                                      for row in sorted(rows, key=lambda item: item["video_id"])]})


def seal_evaluation_labels(root: Path, rows: Sequence[dict[str, Any]], *, layout: TargetLayoutV2,
                           feature_package_id: str, inventory_identity: str,
                           dry_run: bool = False) -> dict[str, Any]:
    """Write and seal the evaluator-only label artifact.

    Building this artifact is NOT revealing the labels. It constructs, validates,
    hashes and seals; it trains nothing, tunes nothing, selects nothing and
    computes no target metric. `target_labels_revealed` stays false until the first
    authorized G8 pass actually reads it to score a model.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from prism_fas.utils.core import sha256_file
    rows = sorted(rows, key=lambda row: row["video_id"])
    if not rows: raise TargetEvalError("the evaluation label artifact cannot be empty")
    if len({row["video_id"] for row in rows}) != len(rows):
        raise TargetEvalError("duplicate video_id in the evaluation label artifact")
    counts = {"videos": len(rows), "live": sum(1 for row in rows if row["label"] == "live"),
              "spoof": sum(1 for row in rows if row["label"] == "spoof")}
    families: dict[str, int] = {}
    for row in rows:
        if row["label"] == "spoof": families[row["attack_family"]] = families.get(row["attack_family"], 0) + 1
    identity = label_artifact_identity(rows, layout_identity=layout.layout_identity())
    lock = {"label_schema_version": LABEL_SCHEMA_VERSION,
            "label_artifact_identity_sha256": identity,
            "layout_rules_version": layout.layout_rules_version,
            "layout_identity_sha256": layout.layout_identity(),
            "adapter_version": layout.adapter_version,
            "source_metadata_provenance": "SiW-Mv2 official directory organization, read once at "
                                          "ingestion by the evaluator-only path",
            "feature_package_id": str(feature_package_id),
            "inventory_identity_sha256": str(inventory_identity),
            "counts": counts, "by_attack_family": dict(sorted(families.items())),
            "join_key": "video_id (opaque siw_<16 hex>)",
            "readable_by": ["G8"], "readable_by_training_or_g7": False,
            "target_labels_revealed": False, "target_label_artifact_built": True,
            "status": "SEALED"}
    if dry_run: return {"lock": lock, "rows": len(rows), "written": []}
    target = Path(root); target.mkdir(parents=True, exist_ok=True)
    parquet = target / "siw_target_labels.parquet"
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows],
                                        schema=pa.schema([("video_id", pa.string()),
                                                          ("label", pa.string()),
                                                          ("attack_family", pa.string())])), parquet)
    lock["label_file_sha256"] = sha256_file(parquet)
    atomic_json_write(target / "TARGET_LABEL_LOCK.json", lock)
    return {"lock": lock, "rows": len(rows), "written": [str(parquet), str(target / "TARGET_LABEL_LOCK.json")]}


# --- live reproduction --------------------------------------------------------

def live_reproduction_audit(*, frozen_package: Path, new_manifest_rows: Sequence[dict[str, Any]],
                            live_video_ids: Iterable[str]) -> dict[str, Any]:
    """The primary acceptance gate.

    Every comparable live frame of the frozen package must reappear in the new
    package with the same `sample_id`, the same `actual_frame_index` and the same
    `crop_sha256`. A mismatch blocks the package: it is investigated, never
    absorbed by relaxing a rule.
    """
    from prism_fas.data.package.manifests import read_manifest
    frozen = {row["sample_id"]: row for row in
              read_manifest(Path(frozen_package) / "manifests" / "target_test_features.parquet")}
    fresh = {row["sample_id"]: row for row in new_manifest_rows}
    live_ids = set(live_video_ids)
    comparable = {sample_id: row for sample_id, row in frozen.items() if row["source_record_id"] in live_ids}
    missing = sorted(set(comparable) - set(fresh))
    crop_mismatch, extra_fields = [], []
    for sample_id, row in sorted(comparable.items()):
        other = fresh.get(sample_id)
        if other is None: continue
        if other["crop_sha256"] != row["crop_sha256"]:
            crop_mismatch.append({"sample_id": sample_id, "frozen": row["crop_sha256"],
                                  "rebuilt": other["crop_sha256"]})
        for key in ("source_record_id", "crop_width", "crop_height"):
            if key in row and key in other and other[key] != row[key]:
                extra_fields.append({"sample_id": sample_id, "field": key,
                                     "frozen": row[key], "rebuilt": other[key]})
    passed = not (missing or crop_mismatch or extra_fields)
    return {"schema_version": TARGET_EVAL_SCHEMA_VERSION, "passed": passed,
            "frozen_live_videos": len(live_ids),
            "frozen_comparable_frames": len(comparable),
            "rebuilt_frames_present": len(comparable) - len(missing),
            "sample_id_mismatches": len(missing), "missing_sample_ids": missing[:20],
            "crop_sha256_mismatches": len(crop_mismatch), "crop_mismatch_examples": crop_mismatch[:20],
            "field_mismatches": len(extra_fields), "field_mismatch_examples": extra_fields[:20],
            "rule": "a comparable frozen live frame must reproduce its sample_id and crop_sha256 "
                    "byte-for-byte under unchanged preprocessing"}


# --- feature-side privacy ------------------------------------------------------

def assert_features_label_free(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A feature row may carry no label, family, taxonomy or private field."""
    leaked = sorted(FORBIDDEN_FEATURE_FIELDS & {key for row in rows for key in row})
    if leaked: raise TargetEvalError(f"the target feature manifest exposes forbidden fields: {leaked}")
    return {"rows": len(rows), "forbidden_fields_present": 0, "labels_present": False}


def assert_no_target_identity(package_root: Path) -> dict[str, Any]:
    """Target priors never receive an identity embedding — asserted, not assumed."""
    from prism_fas.data.package.manifests import read_manifest
    rows = read_manifest(Path(package_root) / "manifests" / "samples.parquet")
    computed = [row["sample_id"] for row in rows if row.get("identity_status") == "computed"]
    if computed:
        raise TargetEvalError(f"{len(computed)} target samples carry an identity embedding")
    return {"samples": len(rows), "target_identity_embeddings": 0,
            "identity_status": sorted({str(row.get("identity_status")) for row in rows})}


# --- plan ----------------------------------------------------------------------

def build_plan(*, raw_root: Path, layout: TargetLayoutV2, audit: InventoryAudit,
               preprocess_config: Any, output_namespace: str, feature_package_id: str,
               label_root: Path, resume: bool) -> dict[str, Any]:
    """The dry-run report. Writes nothing."""
    counts = audit.counts()
    return {"schema_version": TARGET_EVAL_SCHEMA_VERSION, "dry_run": True,
            "inventory_passed": audit.passed, "counts": counts,
            "by_attack_family": audit.by_family(),
            "attack_families": len(audit.by_family()),
            "frames_per_video": int(preprocess_config.frames_per_video),
            "planned_frames": counts["total"] * int(preprocess_config.frames_per_video),
            "output_namespace": output_namespace,
            "feature_package_id": feature_package_id,
            "label_artifact_root": str(label_root),
            "layout_rules_version": layout.layout_rules_version,
            "adapter_version": layout.adapter_version,
            "layout_identity_sha256": layout.layout_identity(),
            "inventory_identity_sha256": audit.identity(),
            "preprocessing_version": preprocess_config.preprocessing_version,
            "preprocessing_config_hash": preprocess_config.config_hash,
            "sampling_version": preprocess_config.sampling_version,
            "output_image_format": preprocess_config.output_image_format,
            "jpeg_quality": preprocess_config.jpeg_quality,
            "crop_output_size": preprocess_config.crop_output_size,
            "crop_padding": preprocess_config.crop_padding,
            "detector": {"variant": preprocess_config.detector.get("model_variant"),
                         "input_size": preprocess_config.scrfd_input_size,
                         "threshold": preprocess_config.detection_threshold},
            "resume": bool(resume),
            "target_identity_embeddings": 0,
            "target_labels_revealed": False,
            "written": []}


def package_identity(*, feature_lock: dict[str, Any], layout: TargetLayoutV2, audit_identity: str,
                     reproduction: dict[str, Any], failure_identity: str) -> dict[str, Any]:
    """The target evaluation package identity.

    Binds the logical inventory, the layout revision, the preprocessing and model
    contracts and the live reproduction RESULT — and no label, no absolute path,
    no host and no clock, so G7 can verify the package without a label existing.
    """
    body = {"schema_version": TARGET_EVAL_SCHEMA_VERSION,
            "package_id": feature_lock["package_id"],
            "feature_content_identity_sha256": feature_lock["content_identity_sha256"],
            "dataset": "siw_mv2",
            "layout_rules_version": layout.layout_rules_version,
            "layout_identity_sha256": layout.layout_identity(),
            "adapter_version": layout.adapter_version,
            "inventory_identity_sha256": audit_identity,
            "declared_videos": int(layout.expected_counts["total"]),
            "declared_live": int(layout.expected_counts["live"]),
            "declared_spoof": int(layout.expected_counts["spoof"]),
            "preprocessing_version": feature_lock.get("preprocessing_version"),
            "preprocessing_config_hash": feature_lock.get("preprocessing_config_hash"),
            "detector_model_sha256": feature_lock.get("detector_model_sha256"),
            "manifest_sha256": feature_lock.get("manifest_sha256"),
            "shards": feature_lock.get("shards"),
            "failure_manifest_identity_sha256": failure_identity,
            "live_reproduction_passed": bool(reproduction["passed"]),
            "live_reproduction_frames": int(reproduction["frozen_comparable_frames"]),
            "target_identity_embeddings": 0,
            "feature_label_separation": {"labels_in_feature_package": False,
                                         "label_artifact_is_separate_tree": True,
                                         "readable_by": ["G8"]},
            "role": "target_evaluation_only"}
    return {**body, "target_package_identity_sha256": stable_json_hash(body)}


def failure_manifest_identity(rows: Sequence[dict[str, Any]]) -> str:
    return stable_json_hash({"schema_version": TARGET_EVAL_SCHEMA_VERSION,
                             "rows": sorted([[str(row.get("sample_id")), str(row.get("source_record_id")),
                                              str(row.get("error_code")), str(row.get("stage"))]
                                             for row in rows])})


def frame_accounting(*, planned: int, successful: int, failed: int) -> dict[str, Any]:
    """`planned = successful + failed`, asserted rather than assumed."""
    reconciled = planned == successful + failed
    if not reconciled:
        raise TargetEvalError(f"frame accounting does not reconcile: planned {planned} != "
                              f"{successful} successful + {failed} failed")
    return {"planned": int(planned), "successful": int(successful), "failed": int(failed),
            "reconciled": reconciled}
