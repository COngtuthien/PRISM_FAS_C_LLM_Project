"""Remote freeze and the single deterministic transport archive for an M8 bank.

A Windows directory download of a many-thousand-file bank failed in M6, so the
validated bank is packed into one uncompressed tar and that single file is moved.
The archive is transport-only: it is never part of the BANK_LOCK identity.
Never imports modal.
"""
from __future__ import annotations
import hashlib, io, json, shutil, tarfile
from pathlib import Path
from typing import Any
from prism_fas.utils.core import atomic_json_write
from .synthetic_shards import TAR_GID, TAR_GNAME, TAR_MODE, TAR_MTIME, TAR_UID, TAR_UNAME

EXPORT_SCHEMA_VERSION = "m8-synthetic-export-v1"
# Everything a validated bank must carry; anything else in the directory is not
# exported.
EXPORT_DIRECTORIES = ("images", "masks", "artifact_maps", "metadata", "manifests", "calibration", "shards")
EXPORT_FILES = ("BANK_LOCK.json", "quality_summary.json", "generation_summary.json", "shards_index.parquet")


class ExportError(RuntimeError):
    """A bank could not be frozen or exported as declared."""


def bank_members(bank_root: Path) -> list[tuple[str, Path]]:
    """Every exported member as `(archive name, source path)`, deterministically
    ordered. The archive name is always `<bank_id>/<relative path>`."""
    root = Path(bank_root)
    bank_id = json.loads((root / "BANK_LOCK.json").read_text(encoding="utf-8"))["bank_id"]
    members: list[tuple[str, Path]] = []
    for name in EXPORT_FILES:
        path = root / name
        if not path.is_file(): raise ExportError(f"{name} is missing from the bank")
        members.append((f"{bank_id}/{name}", path))
    for directory in EXPORT_DIRECTORIES:
        base = root / directory
        if not base.is_dir(): raise ExportError(f"{directory}/ is missing from the bank")
        for path in sorted(base.rglob("*")):
            if path.is_file():
                members.append((f"{bank_id}/{path.relative_to(root).as_posix()}", path))
    members.sort(key=lambda item: item[0])
    return members


def build_archive_bytes(bank_root: Path) -> bytes:
    """One deterministic uncompressed tar of the whole bank."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name, path in bank_members(bank_root):
            payload = path.read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime, info.mode = TAR_MTIME, TAR_MODE
            info.uid, info.gid = TAR_UID, TAR_GID
            info.uname, info.gname = TAR_UNAME, TAR_GNAME
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def export_archive(bank_root: Path, export_root: Path, *, dry_run: bool = False,
                   require_validated: bool = True) -> dict[str, Any]:
    """Write `<export_root>/<bank_id>.tar` and record its size and SHA-256.

    `require_validated=False` exists only to move a **retained failed run** for
    inspection; the archive still carries the bank's real `status`, and freezing
    under the immutable versioned path always requires a validated bank.
    """
    root = Path(bank_root)
    lock = json.loads((root / "BANK_LOCK.json").read_text(encoding="utf-8"))
    if require_validated and lock.get("status") != "validated":
        raise ExportError(f"refusing to export a bank whose status is {lock.get('status')!r}")
    members = bank_members(root)
    payload = build_archive_bytes(root)
    digest = hashlib.sha256(payload).hexdigest()
    target = Path(export_root) / f"{lock['bank_id']}.tar"
    result = {"export_schema_version": EXPORT_SCHEMA_VERSION, "bank_id": lock["bank_id"],
              "bank_status": lock.get("status"),
              "bank_content_identity_sha256": lock["bank_content_identity_sha256"],
              "archive_relative_name": target.name, "archive_bytes": len(payload),
              "archive_sha256": digest, "member_count": len(members),
              "archive_is_identity_bearing": False,
              "archive_excluded_from_bank_lock": True}
    if dry_run: return {**result, "status": "dry_run", "written": []}
    if target.is_file():
        existing = hashlib.sha256(target.read_bytes()).hexdigest()
        if existing == digest: return {**result, "status": "reused", "written": []}
        raise ExportError(f"{target.name} already exists with a different SHA-256")
    from .audit import _atomic_bytes as atomic_bytes_write
    atomic_bytes_write(target, payload)
    return {**result, "status": "created", "written": [target.name]}


def extract_archive(archive_path: Path, destination_root: Path, *, expected_bank_id: str | None = None,
                    overwrite: bool = False) -> dict[str, Any]:
    """Extract one transport archive into `<destination_root>/<bank_id>/`.

    An existing directory holding a *different* bank is never overwritten.
    """
    path = Path(archive_path)
    with tarfile.open(path, mode="r:") as archive:
        names = archive.getnames()
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1: raise ExportError(f"archive must hold exactly one bank directory, found {sorted(roots)}")
        bank_id = roots.pop()
        if expected_bank_id and bank_id != expected_bank_id:
            raise ExportError(f"archive holds {bank_id}, expected {expected_bank_id}")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ExportError(f"refusing to extract unsafe member {name!r}")
        destination = Path(destination_root) / bank_id
        if destination.exists():
            lock_path = destination / "BANK_LOCK.json"
            if lock_path.is_file():
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                member = archive.extractfile(f"{bank_id}/BANK_LOCK.json")
                incoming = json.loads(member.read().decode("utf-8")) if member else {}
                if existing.get("bank_content_identity_sha256") == incoming.get("bank_content_identity_sha256"):
                    return {"status": "already_present", "bank_id": bank_id, "bank_root": str(destination),
                            "members": len(names)}
                if not overwrite: raise ExportError(f"{bank_id} already exists locally with a different identity")
            if overwrite: shutil.rmtree(destination)
        Path(destination_root).mkdir(parents=True, exist_ok=True)
        archive.extractall(Path(destination_root), filter="data")
    return {"status": "extracted", "bank_id": bank_id, "bank_root": str(destination), "members": len(names)}


def freeze_bank(bank_root: Path, frozen_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Copy the validated bank to its versioned immutable path.

    An existing destination with the *same* content identity is reused; an
    existing destination with a different identity is a hard failure.
    """
    root = Path(bank_root)
    lock = json.loads((root / "BANK_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("status") != "validated": raise ExportError("refusing to freeze a bank that is not validated")
    destination = Path(frozen_root) / lock["bank_id"]
    result = {"bank_id": lock["bank_id"], "frozen_relative_name": lock["bank_id"],
              "bank_content_identity_sha256": lock["bank_content_identity_sha256"]}
    if destination.exists():
        existing_lock = destination / "BANK_LOCK.json"
        if not existing_lock.is_file(): raise ExportError(f"{lock['bank_id']} exists at the frozen path without a lock")
        existing = json.loads(existing_lock.read_text(encoding="utf-8"))
        if existing.get("bank_content_identity_sha256") != lock["bank_content_identity_sha256"]:
            raise ExportError(f"{lock['bank_id']} is already frozen with a different content identity")
        return {**result, "status": "reused"}
    if dry_run: return {**result, "status": "dry_run"}
    staging = Path(frozen_root) / f".{lock['bank_id']}.partial"
    if staging.exists(): shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, staging)
    staging.rename(destination)
    return {**result, "status": "frozen"}


def write_export_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    atomic_json_write(Path(path), payload)
    return payload
