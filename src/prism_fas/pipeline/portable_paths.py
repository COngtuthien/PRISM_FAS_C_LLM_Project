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
    preparation cannot call one without this file. Two situations leave it wrong,
    and they are the same defect from a builder's point of view:

    * the file is Git-ignored, so a clone has none at all; and
    * a folder copied from another machine carries one whose roots still name the
      machine it left.

    Both are detected the same way — `project_root` does not name this folder —
    and both are repaired the same way, by deriving every root from the current
    location. A config that already names this folder is authoritative and is
    left exactly as it is, so the machine that authored it keeps working.
    """
    import yaml

    repo = Path(repo).resolve()
    path = repo / "configs" / "paths.local.yaml"
    existing = _local_paths(repo)
    previous = str(existing.get("project_root") or "") if existing else ""

    if existing:
        try:
            same = Path(previous).resolve() == repo
        except (OSError, ValueError):
            same = False
        if same:
            return {"path": path, "action": REUSED, "previous_project_root": previous,
                    "reason": "the existing config names this folder"}

    resolution = resolve(repo)

    def raw_root(name: str) -> str:
        root = resolution.raw.get(name)
        if root and root.present and root.path is not None:
            return root.path.as_posix()
        # Not present anywhere: name the in-folder home, so the builder's own
        # error names the place a collaborator is meant to copy it to.
        return (repo / IN_FOLDER_RAW[name]).as_posix()

    weights = resolution.weights
    document = {
        "workspace_root": repo.parent.as_posix(),
        "project_root": repo.as_posix(),
        "raw_datasets": {name: raw_root(name) for name in sorted(IN_FOLDER_RAW)},
        "model_cache": (weights.path.as_posix() if weights and weights.present
                        and weights.path is not None
                        else (repo / IN_FOLDER_WEIGHTS).as_posix()),
        "work_root": (repo / "data" / "work").as_posix(),
        "processed_root": (repo / "data" / "processed").as_posix(),
        "package_root": (repo / "data" / "packages").as_posix(),
        "runs_root": (repo / "runs").as_posix(),
        "reports_root": (repo / "reports").as_posix(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SYNTHESIZED_HEADER + yaml.safe_dump(document, sort_keys=True),
                    encoding="utf-8")
    return {"path": path, "action": REWRITTEN if existing else WRITTEN,
            "previous_project_root": previous,
            "reason": ("the existing config named a different project root"
                       if existing else "no paths config travelled with the folder")}


# --- what a full scientific run needs, and where it comes from ---------------

#: How each bundle item travels. §20's vocabulary.
MUST_COPY = "MUST_COPY"
REBUILDABLE_FROM_COPIED_RAW = "REBUILDABLE_FROM_COPIED_RAW"
CREATED_DURING_RUN = "CREATED_DURING_RUN"
NOT_REQUIRED_UNTIL_TARGET_STAGE = "NOT_REQUIRED_UNTIL_TARGET_STAGE"


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
            present: bool, needed_from: str, rebuild_hint: str | None = None) -> None:
        items.append({"name": name, "path": relative,
                      "classification": classification, "description": description,
                      "present": present, "needed_from_stage": needed_from,
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
                           ("state", "state"), ("venv", ".venv")):
        add(name, relative, CREATED_DURING_RUN,
            f"the {name} tree, written by the run itself", (repo / relative).exists(),
            "C0")

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


__all__ = ["resolve", "PathResolution", "ResolvedRoot", "bundle_manifest",
           "write_bundle_manifest", "ensure_local_paths", "IN_FOLDER_RAW",
           "IN_FOLDER_WEIGHTS", "DERIVED_ROOTS", "MUST_COPY",
           "REBUILDABLE_FROM_COPIED_RAW", "CREATED_DURING_RUN",
           "NOT_REQUIRED_UNTIL_TARGET_STAGE", "REUSED", "WRITTEN", "REWRITTEN"]
