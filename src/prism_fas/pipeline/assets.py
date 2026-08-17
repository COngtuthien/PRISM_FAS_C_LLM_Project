"""The portable asset manifest: every byte the pipeline needs, and where it is.

The project is meant to be handed over as one folder. This module is what makes
that checkable: it enumerates every external dependency the C4-C13 pipeline
requires, resolves each against the project root, and reports what is present.

Three properties matter and each shapes the code.

**Project-relative.** Every path is expressed relative to the project root, so
moving or copying the folder changes nothing. Roots that genuinely live outside
the folder — the raw datasets, the model cache — are read from
`configs/paths.local.yaml`, which is Git-ignored precisely because it holds
machine-specific absolute paths.

**Identity, never invention.** A hash is recorded only when it is either declared
in a frozen config or computed from bytes on disk. Nothing here fabricates an
expected hash for an artifact it has never seen; an unknown identity is recorded
as unknown.

**Staged.** An item carries which stage first needs it and which intents require
it. The SiW target package is in the manifest, marked required only from C10 and
only for the scientific intent, so a CPU rehearsal is never blocked by — and
never reaches for — the thing it must not touch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-portable-asset-manifest-v1"
MANIFEST_FILE = "PORTABLE_ASSET_MANIFEST.json"

#: Where an item comes from on the target machine.
IN_GIT = "IN_GIT"
IN_FOLDER = "IN_FOLDER"
EXTERNAL_ROOT = "EXTERNAL_ROOT"
GENERATED = "GENERATED_BY_PIPELINE"

READ_ONLY = "READ_ONLY"
WRITE_DESTINATION = "WRITE_DESTINATION"
EVALUATION_ONLY = "EVALUATION_ONLY_NEVER_MOUNTED_ON_TRAINING"


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(root: Path) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    files = [item for item in root.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


@dataclass
class Asset:
    """One external byte dependency."""

    logical_name: str
    expected_path: str
    origin: str
    access: str
    required_stage: str
    required_for_cpu_rehearsal: bool
    required_for_gpu_science: bool
    required_for_real_target_only: bool = False
    identity: str | None = None
    identity_kind: str = ""
    size_bytes: int | None = None
    file_count: int | None = None
    present: bool = False
    how_to_obtain: str = ""
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "expected_path": self.expected_path,
            "origin": self.origin,
            "access": self.access,
            "required_stage": self.required_stage,
            "required_for_cpu_rehearsal": self.required_for_cpu_rehearsal,
            "required_for_gpu_science": self.required_for_gpu_science,
            "required_for_real_target_only": self.required_for_real_target_only,
            "identity": self.identity,
            "identity_kind": self.identity_kind,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "present": self.present,
            "how_to_obtain": self.how_to_obtain,
            "notes": list(self.notes),
        }


def _paths_config(repo: Path) -> dict[str, Any]:
    """Machine-specific roots. Absent on a fresh copy, which is not an error."""
    path = repo / "configs" / "paths.local.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:                                        # noqa: BLE001
        return {}


def build_assets(repo: Path) -> list[Asset]:
    """Enumerate every asset, resolved against this project root."""
    repo = Path(repo)
    paths = _paths_config(repo)
    raw = dict(paths.get("raw_datasets") or {})
    cache = Path(str(paths.get("model_cache", ""))) if paths.get("model_cache") else None
    assets: list[Asset] = []

    def in_git(name: str, relative: str, stage: str, *, rehearsal: bool, science: bool,
               note: str = "") -> None:
        path = repo / relative
        is_dir = path.is_dir()
        count, size = _tree_stats(path) if is_dir else (
            (1, path.stat().st_size) if path.exists() else (0, 0))
        assets.append(Asset(
            logical_name=name, expected_path=relative, origin=IN_GIT, access=READ_ONLY,
            required_stage=stage, required_for_cpu_rehearsal=rehearsal,
            required_for_gpu_science=science, present=path.exists(),
            identity=(_sha256_file(path) if path.is_file() else None),
            identity_kind="sha256_file" if path.is_file() else "",
            size_bytes=size or None, file_count=count or None,
            how_to_obtain="arrives with the project folder",
            notes=(note,) if note else ()))

    # --- frozen scientific inputs that travel with the folder ----------------
    in_git("c3_scientific_recipe_banks", "assets/recipe_banks/c3", "C4",
           rehearsal=True, science=True,
           note="recipes.jsonl is pinned to LF by .gitattributes because its bytes "
                "are hashed")
    in_git("c3_scientific_bank_lock",
           "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json", "C4",
           rehearsal=True, science=True)
    in_git("lr_anchor_decision", "configs/search/lr_anchor_decision.yaml", "C4",
           rehearsal=True, science=True,
           note="the approved learning-rate interpretation; C4 and C7 cannot build a "
                "search plan without it")
    for name, relative, stage in (
            ("gpat_training_config", "configs/synthesis/gpat_m8.yaml", "C4"),
            ("physics_engine_config", "configs/synthesis/physics_m7.yaml", "C5"),
            ("synthetic_bank_config", "configs/synthesis/synthetic_bank_m8.yaml", "C5"),
            ("quality_gate_config", "configs/synthesis/quality_gate_m8.yaml", "C6"),
            ("detector_model_config", "configs/models/m9_detector.yaml", "C7"),
            ("detector_training_config", "configs/train/m9_reference.yaml", "C7"),
            ("experiment_matrix_config", "configs/experiments/m10_matrix.yaml", "C8"),
            ("target_evaluation_config", "configs/evaluation/m10_target.yaml", "C10"),
            ("recipe_ontology", "configs/recipes/ontology_m7.yaml", "C4"),
            ("environment_contract", "configs/environment/environment_contract.yaml", "C0"),
    ):
        in_git(name, relative, stage, rehearsal=True, science=True)

    # --- pinned model weights ------------------------------------------------
    if cache is not None:
        try:
            from prism_fas.detector.pretrained import CONVNEXT_PIN, SIGLIP2_PIN

            siglip_root = cache / SIGLIP2_PIN["local_relpath"]
            files = {name: _sha256_file(siglip_root / name) == spec["sha256"]
                     for name, spec in SIGLIP2_PIN["files"].items()}
            count, size = _tree_stats(siglip_root)
            assets.append(Asset(
                logical_name="siglip2_frozen_global_tower",
                expected_path=str(siglip_root), origin=EXTERNAL_ROOT, access=READ_ONLY,
                required_stage="C7", required_for_cpu_rehearsal=False,
                required_for_gpu_science=True, present=all(files.values()),
                identity=SIGLIP2_PIN["revision"], identity_kind="pinned_revision",
                size_bytes=size or None, file_count=count or None,
                how_to_obtain="re-downloadable at the pinned revision and per-file sha256",
                notes=(f"all {len(files)} pinned files verified: {all(files.values())}",
                       "the rehearsal substitutes a shape-exact fixture tower, so it "
                       "does not require these weights")))

            convnext = next((cache / rel for rel in
                             (CONVNEXT_PIN["local_relpath"],
                              *CONVNEXT_PIN["alternate_relpaths"])
                             if (cache / rel).exists()), None)
            actual = _sha256_file(convnext) if convnext else None
            assets.append(Asset(
                logical_name="convnextv2_atto_local_branch",
                expected_path=str(convnext or cache / CONVNEXT_PIN["local_relpath"]),
                origin=EXTERNAL_ROOT, access=READ_ONLY, required_stage="C7",
                required_for_cpu_rehearsal=False, required_for_gpu_science=True,
                present=actual == CONVNEXT_PIN["weight_sha256"],
                identity=CONVNEXT_PIN["weight_sha256"], identity_kind="sha256_file",
                size_bytes=convnext.stat().st_size if convnext else None,
                how_to_obtain="timm hub at the pinned name and sha256",
                notes=("Track R only; Track G instantiates no local backbone",)))
        except Exception as error:                            # noqa: BLE001
            assets.append(Asset(
                logical_name="pinned_backbones", expected_path=str(cache),
                origin=EXTERNAL_ROOT, access=READ_ONLY, required_stage="C7",
                required_for_cpu_rehearsal=False, required_for_gpu_science=True,
                present=False,
                how_to_obtain=f"pin declarations unreadable: {type(error).__name__}"))

        try:
            import yaml

            gpat = yaml.safe_load(
                (repo / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
            quality = yaml.safe_load(
                (repo / "configs/synthesis/quality_gate_m8.yaml").read_text(encoding="utf-8"))
            for name, candidates, expected, stage, science in (
                    ("adaface_identity_backbone",
                     ("face_identity/pretrained_model/model.pt",),
                     gpat["identity_model"]["weight_sha256"], "C4", True),
                    ("scrfd_face_detector", ("face_detectors/scrfd_10g_bnkps.onnx",),
                     quality["quality_models"]["detector"]["sha256"], "C6", True),
                    ("facexformer_parsing", ("face_geometry/ckpts/model.pt",),
                     quality["quality_models"]["parsing"]["sha256"], "C6", True)):
                found = next((cache / item for item in candidates
                              if (cache / item).exists()), None)
                actual = _sha256_file(found) if found else None
                assets.append(Asset(
                    logical_name=name,
                    expected_path=str(found or cache / candidates[0]),
                    origin=EXTERNAL_ROOT, access=READ_ONLY, required_stage=stage,
                    required_for_cpu_rehearsal=False, required_for_gpu_science=science,
                    present=actual == expected, identity=expected,
                    identity_kind="sha256_file",
                    size_bytes=found.stat().st_size if found else None,
                    how_to_obtain="frozen weight; acquire at the declared sha256"))
        except Exception:                                     # noqa: BLE001
            pass

    # --- raw datasets --------------------------------------------------------
    for key, stage, science, target_only in (("casia_fasd", "C5", True, False),
                                             ("msu_mfsd", "C5", True, False),
                                             ("siw_mv2", "C10", True, True)):
        root = Path(str(raw.get(key, ""))) if raw.get(key) else None
        assets.append(Asset(
            logical_name=f"raw_dataset_{key}",
            expected_path=str(root) if root else f"<{key} root, declare in "
                                                 f"configs/paths.local.yaml>",
            origin=EXTERNAL_ROOT,
            access=EVALUATION_ONLY if target_only else READ_ONLY,
            required_stage=stage, required_for_cpu_rehearsal=False,
            required_for_gpu_science=science,
            required_for_real_target_only=target_only,
            present=bool(root and root.exists()),
            how_to_obtain="licensed dataset; never downloaded or substituted by this "
                          "project",
            notes=(("label files may never be mounted on a training process; only "
                    "label-free features may be mounted read-only for C11",)
                   if target_only else ())))

    # --- derived trees -------------------------------------------------------
    for name, relative, stage, science, how in (
            ("preprocessed_source_data", "data/processed", "C5", True,
             "rebuilt deterministically from the raw roots by the M2 preprocessing "
             "step, or transferred with the folder"),
            ("source_packages", "data/packages", "C4", True,
             "built from the preprocessed tree by the M3 packaging step"),
            ("gpat_pair_plan", "data/packages/gpat_pairs", "C4", True,
             "produced by the frozen source-only pair plan; depends on source_packages"),
            ("target_label_artifact", "data/evaluation_only/prism_target_v2_labels",
             "C12", True,
             "evaluation-only; readable by the isolated C-G8 scorer alone")):
        path = repo / relative
        count, size = _tree_stats(path)
        assets.append(Asset(
            logical_name=name, expected_path=relative, origin=GENERATED,
            access=EVALUATION_ONLY if "label" in name else READ_ONLY,
            required_stage=stage, required_for_cpu_rehearsal=False,
            required_for_gpu_science=science,
            required_for_real_target_only="label" in name,
            present=count > 0, size_bytes=size or None, file_count=count or None,
            how_to_obtain=how))

    # --- write destinations --------------------------------------------------
    for name, relative, stage in (("runs_root", "runs", "C4"),
                                  ("reports_root", "reports", "C4"),
                                  ("state_root", "state", "C0")):
        path = repo / relative
        assets.append(Asset(
            logical_name=name, expected_path=relative, origin=IN_FOLDER,
            access=WRITE_DESTINATION, required_stage=stage,
            required_for_cpu_rehearsal=True, required_for_gpu_science=True,
            present=True,
            how_to_obtain="created by the runner if absent",
            notes=("every artifact stays under the copied folder",)))

    return assets


def load_manifest(repo: Path) -> dict[str, Any]:
    """Build the manifest fresh from the folder as it is right now.

    Deliberately not read from a committed file: a stale manifest that claimed an
    asset was present would defeat the check it exists for. The committed
    PORTABLE_ASSET_MANIFEST.json is a snapshot for the operator to read, not the
    source of truth for the preflight.
    """
    assets = build_assets(repo)
    rows = [item.as_dict() for item in assets]
    material = [{key: row[key] for key in
                 ("logical_name", "expected_path", "origin", "access", "required_stage",
                  "required_for_cpu_rehearsal", "required_for_gpu_science",
                  "required_for_real_target_only", "identity")} for row in rows]
    identity = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_identity": identity,
        "project_relative": True,
        "item_count": len(rows),
        "items": rows,
        "required_for_cpu_rehearsal": sum(1 for row in rows
                                          if row["required_for_cpu_rehearsal"]),
        "required_for_gpu_science": sum(1 for row in rows
                                        if row["required_for_gpu_science"]),
        "present_count": sum(1 for row in rows if row["present"]),
        "policy": {
            "no_invented_hashes": True,
            "no_automatic_dataset_download": True,
            "no_weight_substitution": True,
            "target_assets_required_only_from_c10": True,
        },
    }


def write_manifest(repo: Path) -> Path:
    from prism_fas.pipeline.state import atomic_write_json

    path = Path(repo) / MANIFEST_FILE
    atomic_write_json(path, load_manifest(repo))
    return path


__all__ = ["SCHEMA_VERSION", "MANIFEST_FILE", "IN_GIT", "IN_FOLDER", "EXTERNAL_ROOT",
           "GENERATED", "READ_ONLY", "WRITE_DESTINATION", "EVALUATION_ONLY", "Asset",
           "build_assets", "load_manifest", "write_manifest"]
