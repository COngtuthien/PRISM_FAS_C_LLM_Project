"""Where the big inputs live, resolved so the folder can be copied whole.

`configs/paths.local.yaml` holds absolute machine paths. That works on the
machine that wrote it and nowhere else, which contradicts the workflow this
project is meant to support: copy one folder, run one command.

So every large input now has a project-relative home, and resolution prefers it:

    data/raw/casia_fasd/     data/raw/msu_mfsd/     data/raw/siw_mv2/
    weights/                 (the pinned model cache)

If the in-folder location exists it wins. If it does not, the absolute root from
`paths.local.yaml` is used, which keeps this machine working unchanged. A
collaborator who copies the datasets inside the folder needs no configuration at
all, and one who keeps them elsewhere edits one file.

**Which location was used is operational provenance and never enters a scientific
identity.** That is the property that makes relocation safe: the same data
produces the same identity whether it was read from `D:/Datasets/casia` or from
`./data/raw/casia_fasd`. Identities are computed over content, and
`prism_fas.pipeline.portability` already excludes path-shaped fields from every
identity hash; this module only decides which bytes to open.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Project-relative homes. Copying a dataset here is all a collaborator does.
IN_FOLDER_RAW = {
    "casia_fasd": "data/raw/casia_fasd",
    "msu_mfsd": "data/raw/msu_mfsd",
    "siw_mv2": "data/raw/siw_mv2",
}
IN_FOLDER_WEIGHTS = "weights"

#: Derived trees. Rebuildable from the raw inputs; never copied between machines.
DERIVED_ROOTS = {
    "processed": "data/processed",
    "packages": "data/packages",
    "gpat_pairs": "data/packages/gpat_pairs",
}


@dataclass(frozen=True)
class ResolvedRoot:
    """One input root, and how it was found."""

    name: str
    path: Path | None
    origin: str          # "in_folder" | "paths_local" | "absent"
    present: bool
    declared: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                # Recorded for the operator. Excluded from every identity.
                "path": str(self.path) if self.path else None,
                "origin": self.origin, "present": self.present,
                "declared": self.declared,
                "identity_relevant": False}


@dataclass
class PathResolution:
    """Every large input, resolved once."""

    repo: Path
    raw: dict[str, ResolvedRoot] = field(default_factory=dict)
    weights: ResolvedRoot | None = None
    derived: dict[str, ResolvedRoot] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_datasets": {name: root.as_dict() for name, root in sorted(self.raw.items())},
            "weights": self.weights.as_dict() if self.weights else None,
            "derived": {name: root.as_dict() for name, root in sorted(self.derived.items())},
            "policy": "the in-folder location wins when it exists; otherwise the "
                      "absolute root from configs/paths.local.yaml is used",
            "identity_note": "which location was used is operational provenance. No "
                             "scientific identity depends on it, so the same data "
                             "yields the same identity after the folder moves.",
        }


def _local_paths(repo: Path) -> dict[str, Any]:
    path = repo / "configs" / "paths.local.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except Exception:                                        # noqa: BLE001
        return {}


def _resolve_one(repo: Path, name: str, in_folder: str,
                 declared: str | None) -> ResolvedRoot:
    local = repo / in_folder
    if local.is_dir() and any(local.iterdir()):
        return ResolvedRoot(name, local, "in_folder", True, declared)
    if declared:
        external = Path(declared)
        if external.is_dir():
            return ResolvedRoot(name, external, "paths_local", True, declared)
    return ResolvedRoot(name, None, "absent", False, declared)


def resolve(repo: Path) -> PathResolution:
    """Resolve every large input, preferring the in-folder location."""
    repo = Path(repo).resolve()
    local = _local_paths(repo)
    declared_raw = dict(local.get("raw_datasets") or {})

    resolution = PathResolution(repo=repo)
    for name, relative in IN_FOLDER_RAW.items():
        resolution.raw[name] = _resolve_one(repo, name, relative,
                                            declared_raw.get(name))
    resolution.weights = _resolve_one(repo, "model_cache", IN_FOLDER_WEIGHTS,
                                      local.get("model_cache"))
    for name, relative in DERIVED_ROOTS.items():
        target = repo / relative
        resolution.derived[name] = ResolvedRoot(
            name, target if target.is_dir() else None,
            "in_folder" if target.is_dir() else "absent",
            target.is_dir() and any(target.iterdir()) if target.is_dir() else False,
            relative)
    return resolution


# --- the paths config the canonical builders require -------------------------

#: Written into a synthesized config so a reader knows it is derived, not authored.
SYNTHESIZED_HEADER = (
    "# Generated by prism_fas.pipeline.portable_paths.ensure_local_paths.\n"
    "# Derived from THIS folder's location; safe to delete, regenerated on demand.\n"
    "# Every write root points inside the project. Raw roots prefer the in-folder\n"
    "# copy and fall back to whatever the superseded config declared.\n")

REUSED, WRITTEN, REWRITTEN = "REUSED", "WRITTEN", "REWRITTEN"


def ensure_local_paths(repo: Path) -> dict[str, Any]:
    """Guarantee a `configs/paths.local.yaml` that describes THIS folder.

    The canonical builders take a paths config rather than a repo root, so
    preparation cannot call one without this file. Three situations leave it wrong:

    * the file is Git-ignored, so a clone has none at all;
    * a folder copied from another machine carries one whose roots still name the
      machine it left; and
    * the assets have since been copied INTO the folder while the config still
      points at the machine-specific originals it was written against.

    The third is the one that makes a physically self-contained folder behave as
    though it were not, and it is invisible from `project_root` alone. All three
    are repaired the same way: derive every root from the current location,
    preferring the in-folder copy.

    The override still means something. A root with no in-folder copy keeps
    whatever the superseded config declared, so a machine that legitimately reads
    its corpora from elsewhere keeps working. In-folder wins only where there is
    something in the folder to win with. Write roots are never external.
    """
    import yaml

    repo = Path(repo).resolve()
    path = repo / "configs" / "paths.local.yaml"
    existing = _local_paths(repo)
    previous = str(existing.get("project_root") or "") if existing else ""
    resolution = resolve(repo)

    if existing:
        try:
            same = Path(previous).resolve() == repo
        except (OSError, ValueError):
            same = False
        superseded = _roots_superseded_by_in_folder(repo, existing)
        if same and not superseded:
            return {"path": path, "action": REUSED, "previous_project_root": previous,
                    "reason": "the existing config names this folder and no in-folder "
                              "asset supersedes one of its roots"}
        if same:
            # The config is for this folder, but assets have since been copied
            # INTO the folder and it still points at the machine-specific
            # originals. The in-folder copy wins, or the folder is not portable.
            return _write_config(repo, path, resolution, existing,
                                 action=REWRITTEN, previous=previous,
                                 reason="in-folder assets now supersede external "
                                        f"roots: {', '.join(superseded)}")

    return _write_config(
        repo, path, resolution, existing,
        action=REWRITTEN if existing else WRITTEN, previous=previous,
        reason=("the existing config named a different project root"
                if existing else "no paths config travelled with the folder"))


def _in_folder_root(repo: Path, relative: str) -> Path | None:
    """The in-folder asset root, when it exists and holds something."""
    candidate = repo / relative
    if candidate.is_dir() and any(candidate.iterdir()):
        return candidate
    return None


def _roots_superseded_by_in_folder(repo: Path, existing: dict[str, Any]) -> list[str]:
    """Roots the config aims outside the folder although a copy now lives inside.

    This is what makes a config for THIS folder stale anyway. The assets were
    copied in to make the folder portable; leaving the config aimed at the
    machine-specific originals keeps the runtime bound to them and defeats the
    copy entirely. Write roots are checked too — one pointing outside would put
    this run's outputs on the old machine.
    """
    stale: list[str] = []
    declared_raw = dict(existing.get("raw_datasets") or {})
    for name, relative in IN_FOLDER_RAW.items():
        local = _in_folder_root(repo, relative)
        if local is None:
            continue
        declared = str(declared_raw.get(name) or "")
        try:
            if not declared or Path(declared).resolve() != local:
                stale.append(f"raw_datasets.{name}")
        except (OSError, ValueError):
            stale.append(f"raw_datasets.{name}")

    local_weights = _in_folder_root(repo, IN_FOLDER_WEIGHTS)
    if local_weights is not None:
        declared = str(existing.get("model_cache") or "")
        try:
            if not declared or Path(declared).resolve() != local_weights:
                stale.append("model_cache")
        except (OSError, ValueError):
            stale.append("model_cache")

    for key, relative in (("work_root", "data/work"),
                          ("processed_root", "data/processed"),
                          ("package_root", "data/packages"),
                          ("runs_root", "runs"), ("reports_root", "reports")):
        declared = str(existing.get(key) or "")
        try:
            if not declared or Path(declared).resolve() != (repo / relative).resolve():
                stale.append(key)
        except (OSError, ValueError):
            stale.append(key)
    return stale


def _write_config(repo: Path, path: Path, resolution: PathResolution,
                  existing: dict[str, Any], *, action: str, previous: str,
                  reason: str) -> dict[str, Any]:
    """Write a paths config for this folder, preferring the in-folder assets."""
    import yaml

    def raw_root(name: str) -> str:
        local = _in_folder_root(repo, IN_FOLDER_RAW[name])
        if local is not None:
            return local.as_posix()
        # No in-folder copy: keep whatever the superseded config declared, so a
        # machine that legitimately reads an external root keeps working.
        declared = str((existing.get("raw_datasets") or {}).get(name) or "")
        if declared and Path(declared).is_dir():
            return Path(declared).as_posix()
        # Nowhere at all: name the in-folder home, so the builder's own error
        # names the place a collaborator is meant to copy it to.
        return (repo / IN_FOLDER_RAW[name]).as_posix()

    def weight_root() -> str:
        local = _in_folder_root(repo, IN_FOLDER_WEIGHTS)
        if local is not None:
            return local.as_posix()
        declared = str(existing.get("model_cache") or "")
        if declared and Path(declared).is_dir():
            return Path(declared).as_posix()
        return (repo / IN_FOLDER_WEIGHTS).as_posix()

    document = {
        "workspace_root": repo.parent.as_posix(),
        "project_root": repo.as_posix(),
        "raw_datasets": {name: raw_root(name) for name in sorted(IN_FOLDER_RAW)},
        "model_cache": weight_root(),
        # Every WRITE root is unconditionally inside the project.
        "work_root": (repo / "data" / "work").as_posix(),
        "processed_root": (repo / "data" / "processed").as_posix(),
        "package_root": (repo / "data" / "packages").as_posix(),
        "runs_root": (repo / "runs").as_posix(),
        "reports_root": (repo / "reports").as_posix(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SYNTHESIZED_HEADER + yaml.safe_dump(document, sort_keys=True),
                    encoding="utf-8")
    return {"path": path, "action": action, "previous_project_root": previous,
            "reason": reason}


# --- what a full scientific run needs, and where it comes from ---------------

#: How each bundle item travels. §20's vocabulary.
MUST_COPY = "MUST_COPY"
REBUILDABLE_FROM_COPIED_RAW = "REBUILDABLE_FROM_COPIED_RAW"
CREATED_DURING_RUN = "CREATED_DURING_RUN"
NOT_REQUIRED_UNTIL_TARGET_STAGE = "NOT_REQUIRED_UNTIL_TARGET_STAGE"
#: Present for local convenience, never required by the destination machine.
OPTIONAL_ENGINEERING_ONLY = "OPTIONAL_ENGINEERING_ONLY"


def _stats(path: Path) -> tuple[int | None, int | None]:
    """File count and total bytes for a file or tree; (None, None) if absent."""
    if path.is_file():
        return 1, path.stat().st_size
    if not path.is_dir():
        return None, None
    count = size = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            try:
                size += item.stat().st_size
            except OSError:
                pass
    return count or None, size or None


def bundle_manifest(repo: Path) -> dict[str, Any]:
    """What "copy the whole folder" means, item by item.

    The classification is the useful part. A collaborator does not need to know
    which of these are large, only which ones they must bring, which ones the
    runner will build for them, and which ones do not matter until the target
    stage — at which point a missing one is a legitimate stop rather than a
    surprise mid-run.
    """
    from prism_fas.pipeline.assets import build_assets

    repo = Path(repo).resolve()
    resolution = resolve(repo)
    items: list[dict[str, Any]] = []

    def add(name: str, relative: str, classification: str, description: str,
            present: bool, needed_from: str, rebuild_hint: str | None = None,
            identity_relevant: bool = False) -> None:
        # `present` answers "can this machine resolve it", which was true for the
        # whole period the assets sat outside the folder on an absolute root.
        # `physically_in_folder` is the question that decides portability, and it
        # is measured here rather than inferred from the resolver.
        target = repo / relative
        physical = target.exists() and (
            not target.is_dir() or any(target.iterdir()))
        # The repository row is the whole folder; walking it would re-count the
        # dataset trees this manifest already measures item by item.
        count, size = _stats(target) if relative != "." else (None, None)
        items.append({"name": name, "path": relative,
                      "classification": classification, "description": description,
                      "present": present, "physically_in_folder": physical,
                      "file_count": count or None, "size_bytes": size or None,
                      "needed_from_stage": needed_from,
                      "identity_relevant": identity_relevant,
                      "portability_policy": classification,
                      "rebuild": rebuild_hint})

    add("repository", ".", MUST_COPY,
        "code, configs, frozen C3 recipe banks, locks and all committed evidence",
        True, "C0")

    for name, root in sorted(resolution.raw.items()):
        target_only = name == "siw_mv2"
        add(f"raw_{name}", IN_FOLDER_RAW[name],
            NOT_REQUIRED_UNTIL_TARGET_STAGE if target_only else MUST_COPY,
            f"the {name.upper().replace('_', '-')} raw dataset"
            + (" (held-out target; not opened before C10)" if target_only else ""),
            root.present, "C10" if target_only else "C4",
            "copy the dataset into this path, or point configs/paths.local.yaml at it")

    add("weights", IN_FOLDER_WEIGHTS, MUST_COPY,
        "the pinned SigLIP2, ConvNeXt, AdaFace, SCRFD and FaceXFormer weights",
        bool(resolution.weights and resolution.weights.present), "C4",
        "copy the model cache into ./weights, or point configs/paths.local.yaml at it")

    for name, relative in DERIVED_ROOTS.items():
        root = resolution.derived[name]
        add(f"derived_{name}", relative, REBUILDABLE_FROM_COPIED_RAW,
            f"the derived {name} tree",
            root.present, "C4",
            "built automatically by `python train.py` from the raw datasets; "
            "never copied between machines")

    for name, relative in (("runs", "runs"), ("reports", "reports"),
                           ("state", "state")):
        add(name, relative, CREATED_DURING_RUN,
            f"the {name} tree, written by the run itself", (repo / relative).exists(),
            "C0")

    # Deliberately not CREATED_DURING_RUN: an interpreter tree is machine, OS and
    # CUDA specific, so copying it across machines is worse than useless. The
    # bootstrap builds a fresh one on the destination.
    add("venv", ".venv", OPTIONAL_ENGINEERING_ONLY,
        "this machine's interpreter; EXCLUDE from the transfer",
        (repo / ".venv").exists(), "C0",
        "recreated automatically by bootstrap.py on the destination machine")

    # The pinned weight files, individually, so a partial copy is detectable.
    # `build_assets` returns dataclasses; only the weight rows are relevant here.
    try:
        assets = [
            {"logical_name": item.logical_name, "path": item.expected_path,
             "present": item.present, "identity": item.identity,
             "required_stage": item.required_stage}
            for item in build_assets(repo)
            if "weight" in item.logical_name or item.origin == "pinned_weight"]
    except Exception:                                        # noqa: BLE001
        assets = []

    required_now = [item for item in items
                    if item["classification"] == MUST_COPY and not item["present"]]
    rebuildable_missing = [item for item in items
                           if item["classification"] == REBUILDABLE_FROM_COPIED_RAW
                           and not item["present"]]
    raw_for_rebuild_present = all(
        resolution.raw[name].present for name in ("casia_fasd", "msu_mfsd"))

    ready = not required_now and (not rebuildable_missing or raw_for_rebuild_present)
    return {
        "schema_version": "prism-portable-bundle-v1",
        "items": items,
        "weight_files": assets,
        "resolution": resolution.as_dict(),
        "bundle_ready_for_full": "YES" if ready else "NO",
        # Deduplicated and ordered: the raw datasets appear both as MUST_COPY
        # items and as the source a REBUILDABLE item needs, and naming them twice
        # reads as two separate problems.
        "blockers": sorted({item["path"] for item in required_now}
                           | (set() if raw_for_rebuild_present or not rebuildable_missing
                              else {"data/raw/casia_fasd", "data/raw/msu_mfsd"})),
        "counts": {
            "must_copy": sum(1 for item in items if item["classification"] == MUST_COPY),
            "rebuildable": sum(1 for item in items
                               if item["classification"] == REBUILDABLE_FROM_COPIED_RAW),
            "created_during_run": sum(1 for item in items
                                      if item["classification"] == CREATED_DURING_RUN),
            "target_stage_only": sum(
                1 for item in items
                if item["classification"] == NOT_REQUIRED_UNTIL_TARGET_STAGE),
            "optional_engineering_only": sum(
                1 for item in items
                if item["classification"] == OPTIONAL_ENGINEERING_ONLY),
            "unclassified": sum(
                1 for item in items if item["classification"] not in
                (MUST_COPY, REBUILDABLE_FROM_COPIED_RAW, CREATED_DURING_RUN,
                 NOT_REQUIRED_UNTIL_TARGET_STAGE, OPTIONAL_ENGINEERING_ONLY)),
        },
        "note": "a REBUILDABLE item absent with its raw source present is not a "
                "blocker: `python train.py` builds it before C4 and resumes an "
                "interrupted build rather than restarting it",
    }


def write_bundle_manifest(repo: Path) -> Path:
    from prism_fas.pipeline.state import atomic_write_json

    repo = Path(repo).resolve()
    path = repo / "PORTABLE_BUNDLE_MANIFEST.json"
    atomic_write_json(path, bundle_manifest(repo))
    return path


TRANSFER_MANIFEST = "PORTABLE_TRANSFER_MANIFEST.json"

#: Hashed individually. Small, identity-critical, and the files whose corruption
#: would be silent: a truncated weight still loads far enough to waste a GPU day.
_CRITICAL_HASH_TARGETS = (
    "configs/search/lr_anchor_decision.yaml",
    "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
    "reports/c3/C3_BANK_LOCK.json",
    "reports/c3/v15_selection_contract/C3_BANK_CONTRACT_LOCK.json",
    "reports/c0/VERSION_B_INTEGRITY_SNAPSHOT.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    """A dataset's identity without hashing every frame.

    CASIA alone is 123k files. Hashing all of them on the destination would cost
    more than the transfer, so the fingerprint is a digest over the sorted
    (relative path, size) pairs. It catches a truncated, partial or reordered
    copy — the realistic transfer failures — and says plainly that it is not a
    byte-level claim.
    """
    digest = hashlib.sha256()
    count = size = 0
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        stat = item.stat()
        digest.update(relative.encode())
        digest.update(str(stat.st_size).encode())
        count += 1
        size += stat.st_size
    return {"files": count, "bytes": size, "manifest_sha256": digest.hexdigest(),
            "claim": "path+size manifest digest, not per-file content hashes"}


def transfer_manifest(repo: Path) -> dict[str, Any]:
    """What the destination machine should check before it starts training."""
    from prism_fas.pipeline.assets import build_assets

    repo = Path(repo).resolve()
    weights: list[dict[str, Any]] = []
    for asset in build_assets(repo):
        # IN_GIT assets carry a repo-relative path, EXTERNAL_ROOT ones an absolute
        # path that is now inside the folder. Normalise before comparing.
        path = Path(asset.expected_path)
        if not path.is_absolute():
            path = repo / path
        if asset.identity_kind != "sha256_file" or not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(repo).as_posix()
        except ValueError:
            # Still outside the folder: record it as such rather than crashing,
            # because that is exactly the condition this manifest must surface.
            relative = str(path)
        weights.append({"logical_name": asset.logical_name,
                        "path": relative,
                        "inside_project": not Path(relative).is_absolute(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "expected_sha256": asset.identity,
                        "matches_frozen_pin": _sha256(path) == asset.identity,
                        "required_stage": asset.required_stage})

    siglip = repo / IN_FOLDER_WEIGHTS / "pretrained/m9/siglip2"
    if siglip.is_dir():
        for item in sorted(siglip.rglob("*")):
            if item.is_file():
                weights.append({"logical_name": "siglip2_frozen_global_tower",
                                "path": item.relative_to(repo).as_posix(),
                                "inside_project": True,
                                "bytes": item.stat().st_size,
                                "sha256": _sha256(item),
                                "expected_sha256": None,
                                "matches_frozen_pin": None,
                                "required_stage": "C7"})

    datasets = {}
    for name, relative in IN_FOLDER_RAW.items():
        root = repo / relative
        datasets[name] = ({"path": relative, "present": False}
                          if not root.is_dir() else
                          {"path": relative, "present": True, **_tree_fingerprint(root)})

    critical = {}
    for relative in _CRITICAL_HASH_TARGETS:
        path = repo / relative
        critical[relative] = _sha256(path) if path.is_file() else None

    banks = {}
    for arm in ("llm", "rnd", "det"):
        root = repo / "assets" / "recipe_banks" / "c3" / arm
        if root.is_dir():
            banks[arm] = _tree_fingerprint(root)

    return {
        "schema_version": "prism-portable-transfer-v1",
        "generated_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "verify a copied folder on the destination machine BEFORE training",
        "how_to_verify": "python -m prism_fas.cli.main verify-transfer  (or re-run "
                         "transfer_manifest() and compare this file field by field)",
        "source_project_root_is_provenance_only": True,
        "datasets": datasets,
        "frozen_weights": weights,
        "c3_recipe_banks": banks,
        "critical_files": critical,
        "excluded_from_transfer": [
            ".venv", "__pycache__", ".pytest_cache",
            "data/raw/msu_mfsd/MSU-MFSD-Publish.zip.0*  (split archives; the "
            "extracted tree beside them is what the adapter reads)"],
        "note": "dataset entries are path+size manifest digests, not per-frame "
                "content hashes; frozen weights and critical files are full SHA256.",
    }


def write_transfer_manifest(repo: Path) -> Path:
    from prism_fas.pipeline.state import atomic_write_json

    repo = Path(repo).resolve()
    path = repo / TRANSFER_MANIFEST
    atomic_write_json(path, transfer_manifest(repo))
    return path


__all__ = ["resolve", "PathResolution", "ResolvedRoot", "bundle_manifest",
           "write_bundle_manifest", "ensure_local_paths", "IN_FOLDER_RAW",
           "IN_FOLDER_WEIGHTS", "DERIVED_ROOTS", "MUST_COPY",
           "REBUILDABLE_FROM_COPIED_RAW", "CREATED_DURING_RUN",
           "NOT_REQUIRED_UNTIL_TARGET_STAGE", "REUSED", "WRITTEN", "REWRITTEN"]
