"""The GPU-handoff inventory: everything a later full run needs, and where it is.

The scientific pipeline will be executed somewhere else, on a machine that has
this repository and nothing else. This module answers the question that machine
will ask first: *what do I need that is not in the clone, and how do I know I got
the right bytes?*

Every entry therefore carries four things that a bare path does not: whether the
artifact is inside Git or has to arrive some other way, what identity it must
hash to, which stage first needs it, and whether it may be written or only read.
The last one matters more than it looks — the SiW target package is listed here,
and it is listed as read-only, needed no earlier than C10, and forbidden to every
training process. An inventory that merely said "you will need SiW" would be an
invitation to mount it in the wrong place.

Two things this module deliberately does not do. It never copies a dataset —
inventories describe, they do not stage. And it never records a secret: only the
presence of a credential is ever reported, never its value.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "prism-gpu-handoff-inventory-v1"

#: Where an artifact comes from on the target machine.
IN_GIT = "IN_GIT"                      # arrives with the clone
EXTERNAL = "EXTERNAL_REQUIRED"         # must be supplied; not in the clone
GENERATED = "GENERATED_BY_PIPELINE"    # the run produces it
ABSENT = "NOT_YET_PRODUCED"            # a destination that does not exist yet

#: Access policy. Enforced elsewhere; recorded here so the operator can see it.
READ_ONLY = "READ_ONLY"
READ_WRITE = "READ_WRITE"
WRITE_DESTINATION = "WRITE_DESTINATION"
EVALUATION_ONLY = "EVALUATION_ONLY_NEVER_MOUNTED_ON_TRAINING"


class HandoffError(RuntimeError):
    """The inventory cannot be built from repository evidence."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(root: Path) -> tuple[str, int, int]:
    """A directory identity: hash of (relative path, file hash) pairs, sorted.

    Sorted and relative so the same tree hashes the same on Windows and Linux,
    which is the entire point of putting it in a handoff document.
    """
    entries: list[tuple[str, str]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        entries.append((path.relative_to(root).as_posix(), _sha256_file(path)))
        total += path.stat().st_size
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(entries), total


@dataclass(frozen=True)
class InventoryItem:
    """One artifact the future full run depends on."""

    logical_name: str
    expected_path: str
    origin: str
    access: str
    required_by_stages: tuple[str, ...]
    description: str = ""
    identity: str | None = None
    identity_kind: str = ""
    size_bytes: int | None = None
    file_count: int | None = None
    present: bool = False
    verification: str = ""
    notes: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        """An external input that is absent blocks the stage that needs it."""
        return self.origin == EXTERNAL and not self.present

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "expected_path": self.expected_path,
            "origin": self.origin,
            "stored_in_git": self.origin == IN_GIT,
            "stored_externally": self.origin == EXTERNAL,
            "access": self.access,
            "required_by_stages": list(self.required_by_stages),
            "description": self.description,
            "identity": self.identity,
            "identity_kind": self.identity_kind,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "present_on_this_machine": self.present,
            "blocks_future_full_run": self.blocking,
            "verification": self.verification,
            "notes": list(self.notes),
        }


def _file_item(repo: Path, *, logical_name: str, relative: str, origin: str,
               access: str, stages: Sequence[str], description: str,
               verification: str, notes: Sequence[str] = ()) -> InventoryItem:
    path = repo / relative
    present = path.exists() and path.is_file()
    return InventoryItem(
        logical_name=logical_name, expected_path=relative, origin=origin, access=access,
        required_by_stages=tuple(stages), description=description,
        identity=_sha256_file(path) if present else None,
        identity_kind="sha256_file" if present else "",
        size_bytes=path.stat().st_size if present else None,
        present=present, verification=verification, notes=tuple(notes))


def _tree_item(repo: Path, *, logical_name: str, relative: str, origin: str,
               access: str, stages: Sequence[str], description: str,
               verification: str, notes: Sequence[str] = ()) -> InventoryItem:
    path = repo / relative
    present = path.exists() and path.is_dir()
    identity, count, size = _sha256_tree(path) if present else (None, None, None)
    return InventoryItem(
        logical_name=logical_name, expected_path=relative, origin=origin, access=access,
        required_by_stages=tuple(stages), description=description,
        identity=identity, identity_kind="sha256_tree" if present else "",
        size_bytes=size, file_count=count, present=present,
        verification=verification, notes=tuple(notes))


def _external_item(*, logical_name: str, expected_path: str, access: str,
                   stages: Sequence[str], description: str, verification: str,
                   present: bool, notes: Sequence[str] = ()) -> InventoryItem:
    return InventoryItem(
        logical_name=logical_name, expected_path=expected_path, origin=EXTERNAL,
        access=access, required_by_stages=tuple(stages), description=description,
        present=present, verification=verification, notes=tuple(notes))


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_inventory(repo: Path) -> list[InventoryItem]:
    """Every input, destination and dependency the full pipeline will need.

    Built by looking, not by declaring: identities are hashed off disk where the
    artifact exists, and an absent artifact is recorded as absent rather than
    omitted. A short inventory would be a worse handoff than an honest one with
    holes in it.
    """
    repo = Path(repo)
    paths = _read_yaml(repo / "configs/paths.local.yaml")
    raw = dict(paths.get("raw_datasets") or {})
    items: list[InventoryItem] = []

    # --- frozen scientific inputs that travel with the clone ------------------
    items.append(_tree_item(
        repo, logical_name="c3_scientific_recipe_banks",
        relative="assets/recipe_banks/c3", origin=IN_GIT, access=READ_ONLY,
        stages=("C5", "C6", "C7", "C8"),
        description="the frozen 256-recipe banks for RND, DET and LLM produced by C3",
        verification="re-derive each arm's bank_identity from C3_BANK.json's own declared "
                     "bank_identity_material and compare to C3_SCIENTIFIC_BANK_LOCK.json",
        notes=("recipes.jsonl is pinned to LF by .gitattributes because its bytes are "
               "hashed; a CRLF checkout would produce a different identity",)))
    items.append(_file_item(
        repo, logical_name="c3_scientific_bank_lock",
        relative="reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
        origin=IN_GIT, access=READ_ONLY, stages=("C5", "C6", "C7", "C8", "C9"),
        description="the governing lock over the three frozen scientific banks",
        verification="recompute lock_identity from lock_identity_material"))
    items.append(_file_item(
        repo, logical_name="c3_generation_bank_lock", relative="reports/c3/C3_BANK_LOCK.json",
        origin=IN_GIT, access=READ_ONLY, stages=("C5",),
        description="the preliminary C3 bank lock; superseded but immutable evidence",
        verification="sha256 must equal the value recorded in PROJECT_STATE"))
    items.append(_tree_item(
        repo, logical_name="c3_raw_candidate_archives",
        relative="reports/c3/live/raw_responses", origin=IN_GIT, access=READ_ONLY,
        stages=("C13",),
        description="the 12 immutable raw provider responses behind the LLM arm",
        verification="sha256 of each archive's raw_response equals the archive_identity "
                     "recorded in C3_LIVE_GENERATION_STATE.json"))
    items.append(_file_item(
        repo, logical_name="governing_spec",
        relative="docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_FullPipeline_Spec_2026.docx",
        origin=IN_GIT, access=READ_ONLY, stages=("C0",),
        description="the governing v1.5 contract",
        verification="sha256 must equal ad8495f2576607546ff8c3bd4f47991197cbb3802265a"
                     "599d1808aa1a97066e5"))

    # --- configuration -------------------------------------------------------
    for name, relative, stages in (
            ("gpat_training_config", "configs/synthesis/gpat_m8.yaml", ("C4",)),
            ("physics_engine_config", "configs/synthesis/physics_m7.yaml", ("C5",)),
            ("synthetic_bank_config", "configs/synthesis/synthetic_bank_m8.yaml", ("C5", "C6")),
            ("quality_gate_config", "configs/synthesis/quality_gate_m8.yaml", ("C6",)),
            ("detector_model_config", "configs/models/m9_detector.yaml", ("C7", "C8")),
            ("detector_training_config", "configs/train/m9_reference.yaml", ("C7", "C8")),
            ("experiment_matrix_config", "configs/experiments/m10_matrix.yaml", ("C8", "C11")),
            ("target_evaluation_config", "configs/evaluation/m10_target.yaml", ("C10", "C11", "C12")),
            ("recipe_ontology", "configs/recipes/ontology_m7.yaml", ("C5", "C6")),
            ("execution_profile_full", "configs/execution/full.yaml", ("C0",)),
    ):
        items.append(_file_item(
            repo, logical_name=name, relative=relative, origin=IN_GIT,
            access=READ_ONLY, stages=stages,
            description=f"frozen configuration consumed by {', '.join(stages)}",
            verification="sha256 recorded here; the loader recomputes the config identity "
                         "that enters every run manifest"))

    # --- external datasets and model weights ---------------------------------
    for key, stages, description in (
            ("casia_fasd", ("C5", "C7", "C8"), "CASIA-FASD source domain"),
            ("msu_mfsd", ("C5", "C7", "C8"), "MSU-MFSD source domain")):
        location = str(raw.get(key, ""))
        items.append(_external_item(
            logical_name=f"raw_dataset_{key}", expected_path=location or f"<{key} root>",
            access=READ_ONLY, stages=stages, description=description,
            present=bool(location) and Path(location).exists(),
            verification="the preprocessing manifest identity, not the raw bytes; raw roots "
                         "are never copied into the repository or uploaded anywhere",
            notes=("declared in configs/paths.local.yaml, which is Git-ignored because it "
                   "holds machine-specific absolute paths",)))

    siw_root = str(raw.get("siw_mv2", ""))
    items.append(_external_item(
        logical_name="target_feature_package_siw_mv2_v2",
        expected_path=siw_root or "<SiW-Mv2 root>", access=EVALUATION_ONLY,
        stages=("C10", "C11", "C12"),
        description="the fixed P3 held-out target feature package",
        present=bool(siw_root) and Path(siw_root).exists(),
        verification="feature package identity verified at C10 against the frozen target "
                     "lock; label files are verified only inside the isolated C12 scorer",
        notes=("label-free features may be mounted read-only for C11 inference",
               "label files MUST NOT be mounted on any training process or volume, and no "
               "stage before C10 may resolve this root at all",
               "not required for engineering readiness; needed only from C10 onward")))
    # The pinned backbones, resolved through their canonical pin declarations and
    # verified file by file. Reporting the cache root's existence alone would say
    # "present" for a cache holding the wrong revision.
    cache_root = Path(str(paths.get("model_cache", "")))
    backbones: list[dict[str, Any]] = []
    try:
        from prism_fas.detector.pretrained import CONVNEXT_PIN, SIGLIP2_PIN

        siglip_root = cache_root / SIGLIP2_PIN["local_relpath"]
        siglip_files = {
            name: _sha256_file(siglip_root / name) == spec["sha256"]
            for name, spec in SIGLIP2_PIN["files"].items()}
        backbones.append({"name": "siglip2", "verified": all(siglip_files.values()),
                          "files": siglip_files, "root": str(siglip_root)})
        convnext = next((cache_root / rel for rel in
                         (CONVNEXT_PIN["local_relpath"], *CONVNEXT_PIN["alternate_relpaths"])
                         if (cache_root / rel).exists()), None)
        backbones.append({
            "name": "convnextv2_atto",
            "verified": bool(convnext) and _sha256_file(convnext) == CONVNEXT_PIN["weight_sha256"],
            "root": str(convnext) if convnext else None})
    except Exception as error:  # a pin that cannot be read is itself the finding
        backbones.append({"name": "resolution_failed", "verified": False,
                          "error": f"{type(error).__name__}: {error}"})

    items.append(_external_item(
        logical_name="model_cache", expected_path=str(cache_root or "<model cache>"),
        access=READ_ONLY, stages=("C7", "C8", "C11"),
        description="pinned backbone weights: frozen SigLIP2 image tower and ConvNeXt V2 Atto",
        present=bool(backbones) and all(item.get("verified") for item in backbones),
        verification="every pinned file is hashed against the sha256 in its pin "
                     "declaration; a mismatch is a hard error, not a warning",
        notes=("the engineering smoke substitutes a fixture tower and therefore does NOT "
               "exercise these weights; the full profile requires the pinned identities",
               *(f"{item['name']}: verified={item.get('verified')}" for item in backbones))))

    identity_model = _read_yaml(repo / "configs/synthesis/gpat_m8.yaml").get("identity_model", {})
    if identity_model:
        # Resolved against the declared model cache and verified by hash rather
        # than assumed absent. An inventory that reports a present, verified
        # dependency as missing sends the operator to re-download 174 MB they
        # already have, and — worse — casts doubt on the entries that really are
        # missing.
        expected = str(identity_model.get("weight_sha256", ""))
        cache = Path(str(paths.get("model_cache", "")))
        candidates = ("face_identity/pretrained_model/model.pt",
                      "adaface_ir50/model.pt", "face_identity/model.pt")
        resolved = next((cache / name for name in candidates if (cache / name).exists()),
                        None)
        actual = _sha256_file(resolved) if resolved else None
        items.append(_external_item(
            logical_name="gpat_identity_model_adaface",
            expected_path=str(resolved) if resolved else f"{cache}/{candidates[0]}",
            access=READ_ONLY, stages=("C4", "C6"),
            description="frozen AdaFace identity backbone used by the GPAT identity loss",
            present=bool(actual) and actual == expected,
            verification=f"weight sha256 {expected or 'unpinned'} at revision "
                         f"{identity_model.get('revision', 'unpinned')}; "
                         f"{'VERIFIED on this machine' if actual == expected else 'not verified here'}",
            notes=("resolved from the declared model cache; never trained",
                   f"measured sha256: {actual or 'file not found'}")))

    # --- pipeline-produced inputs and destinations ---------------------------
    for name, relative, stages, description in (
            ("preprocessed_source_data", "data/processed", ("C5", "C7", "C8"),
             "preprocessed CASIA/MSU frames and metadata"),
            ("source_packages", "data/packages", ("C4", "C5", "C7", "C8"),
             "built source packages: shards, manifests and priors"),
    ):
        path = repo / relative
        present = path.exists() and any(path.rglob("*")) if path.exists() else False
        items.append(InventoryItem(
            logical_name=name, expected_path=relative,
            origin=GENERATED if present else EXTERNAL, access=READ_ONLY,
            required_by_stages=tuple(stages), description=description, present=present,
            verification="package identity recorded in the package manifest and pinned into "
                         "every downstream run identity",
            notes=("large and Git-ignored; rebuilt on the target machine from the raw roots "
                   "by the same preprocessing configs, or transferred out of band",)))

    for name, relative, stages, description in (
            ("gpat_checkpoint_destination", "runs/full/c4", ("C4", "C5"),
             "final GPAT checkpoint and its lock"),
            ("detector_checkpoint_destination", "runs/full/c8", ("C8", "C9", "C11"),
             "per method/config/seed detector checkpoints and calibration artifacts"),
            ("full_report_destination", "reports/full", ("C4", "C13"),
             "per-milestone full-profile evidence: raw, metrics, audits, selection, locks"),
    ):
        path = repo / relative
        items.append(InventoryItem(
            logical_name=name, expected_path=relative, origin=GENERATED,
            access=WRITE_DESTINATION, required_by_stages=tuple(stages),
            description=description, present=path.exists(),
            verification="written atomically; every artifact serializes execution_profile "
                         "and scientific_eligible, and every run row lands in "
                         "state/MASTER_RUN_INDEX.json",
            notes=("empty on a fresh clone; the full run creates it",)))

    return items


def credential_presence() -> dict[str, str]:
    """Presence only. A key's value is never read, logged or serialized."""
    import os

    return {name: ("PRESENT" if os.environ.get(name) else "MISSING")
            for name in ("GEMINI_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")}


@dataclass
class HandoffReport:
    """The engineering-readiness handoff document."""

    repo: Path
    items: list[InventoryItem]
    generated_at_utc: str = field(default_factory=_utc)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_items(self) -> list[InventoryItem]:
        return [item for item in self.items if item.blocking]

    def inventory_identity(self) -> str:
        """Identity over the inventory's content, excluding the clock."""
        payload = [item.as_dict() for item in self.items]
        for row in payload:
            row.pop("present_on_this_machine", None)
            row.pop("blocks_future_full_run", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        from prism_fas.pipeline.portability import KNOWN_BACKENDS

        by_origin: dict[str, int] = {}
        for item in self.items:
            by_origin[item.origin] = by_origin.get(item.origin, 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "inventory_identity": self.inventory_identity(),
            "item_count": len(self.items),
            "items_by_origin": by_origin,
            "items": [item.as_dict() for item in self.items],
            "external_inputs_absent_here": [item.logical_name for item in self.blocking_items],
            "credentials": credential_presence(),
            "credential_policy": "presence only; no key is ever printed, logged or committed",
            "supported_backends": {name: profile.as_dict()
                                   for name, profile in KNOWN_BACKENDS.items()},
            "backend_contract": (
                "scientific identities must not depend on the backend, workspace, GPU model, "
                "instance id, physical batch size or filesystem root. Backend information is "
                "operational provenance only (L.12)"),
            "resume_contract": (
                "--resume is identity-aware: a completed unit is skipped only after its "
                "parent identities, config identity, content hash and acceptance state all "
                "validate. A valid frozen C3 archive or GPAT checkpoint is never regenerated "
                "because the orchestrator restarted (L.11)"),
            **self.extra,
        }


def build_report(repo: Path, **extra: Any) -> HandoffReport:
    return HandoffReport(repo=Path(repo), items=build_inventory(repo), extra=dict(extra))


__all__ = ["SCHEMA_VERSION", "IN_GIT", "EXTERNAL", "GENERATED", "ABSENT", "READ_ONLY",
           "READ_WRITE", "WRITE_DESTINATION", "EVALUATION_ONLY", "HandoffError",
           "InventoryItem", "build_inventory", "credential_presence", "HandoffReport",
           "build_report"]
