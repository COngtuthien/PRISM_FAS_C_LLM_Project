"""Deterministic uncompressed tar shards for the accepted M8 bank.

Byte-for-byte reproducibility is the whole point: member order, member metadata
and mtime are all fixed, so two builds of the same accepted set produce identical
shard bytes and identical shard hashes. Never imports modal.
"""
from __future__ import annotations
import hashlib, io, tarfile
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq

SHARD_SCHEMA_VERSION = "m8-synthetic-shards-v1"
SHARD_PREFIX = "synthetic"
MAX_SAMPLES_PER_SHARD = 500
# Fixed member metadata: nothing about the building machine or the wall clock may
# reach the bytes.
TAR_UID, TAR_GID, TAR_UNAME, TAR_GNAME, TAR_MODE, TAR_MTIME = 0, 0, "", "", 0o644, 0
MEMBER_SUFFIXES = (".png", ".mask.png", ".npz", ".json")
ALLOWED_DATASETS = ("casia_fasd", "msu_mfsd")
ROUTES = ("physics", "gpat")


class ShardError(RuntimeError):
    """A shard could not be built or validated as declared."""


def shard_name(index: int) -> str:
    return f"{SHARD_PREFIX}-{index:05d}.tar"


def member_names(synthetic_id: str) -> tuple[str, ...]:
    return tuple(f"{synthetic_id}{suffix}" for suffix in MEMBER_SUFFIXES)


def _member_sources(bank_root: Path, synthetic_id: str) -> list[tuple[str, Path]]:
    root = Path(bank_root)
    return [(f"{synthetic_id}.png", root / "images" / f"{synthetic_id}.png"),
            (f"{synthetic_id}.mask.png", root / "masks" / f"{synthetic_id}.png"),
            (f"{synthetic_id}.npz", root / "artifact_maps" / f"{synthetic_id}.npz"),
            (f"{synthetic_id}.json", root / "metadata" / f"{synthetic_id}.json")]


def build_shard_bytes(bank_root: Path, synthetic_ids: list[str]) -> bytes:
    """One uncompressed tar. Members are emitted in `(synthetic_id, suffix)` order
    with fixed uid/gid/uname/gname/mode/mtime."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for synthetic_id in sorted(synthetic_ids):
            for name, path in _member_sources(bank_root, synthetic_id):
                if not path.is_file(): raise ShardError(f"accepted member {name} is missing from the bank")
                payload = path.read_bytes()
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                info.mtime = TAR_MTIME
                info.mode = TAR_MODE
                info.uid, info.gid = TAR_UID, TAR_GID
                info.uname, info.gname = TAR_UNAME, TAR_GNAME
                info.type = tarfile.REGTYPE
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def plan_shards(accepted: list[dict[str, Any]], *, max_samples: int = MAX_SAMPLES_PER_SHARD) -> list[list[dict[str, Any]]]:
    """Accepted rows only, sorted by `synthetic_id`, at most `max_samples` each."""
    if max_samples <= 0: raise ShardError("a shard must hold at least one sample")
    rows = sorted(accepted, key=lambda row: row["synthetic_id"])
    return [rows[start:start + max_samples] for start in range(0, len(rows), max_samples)]


_INDEX_FIELDS = [("shard_name", pa.string()), ("row_count", pa.int32()), ("byte_size", pa.int64()),
                 ("sha256", pa.string()), ("first_synthetic_id", pa.string()), ("last_synthetic_id", pa.string()),
                 ("physics_count", pa.int32()), ("gpat_count", pa.int32()),
                 ("live_casia_fasd_count", pa.int32()), ("live_msu_mfsd_count", pa.int32())]


def build_shards(bank_root: Path, accepted: list[dict[str, Any]], *,
                 max_samples: int = MAX_SAMPLES_PER_SHARD) -> list[dict[str, Any]]:
    """Write every shard under `<bank_root>/shards/` and return the index rows."""
    root = Path(bank_root)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for position, group in enumerate(plan_shards(accepted, max_samples=max_samples)):
        ids = [row["synthetic_id"] for row in group]
        payload = build_shard_bytes(root, ids)
        name = shard_name(position)
        from .audit import _atomic_bytes as atomic_bytes_write
        atomic_bytes_write(root / "shards" / name, payload)
        index.append({"shard_name": name, "row_count": len(group), "byte_size": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "first_synthetic_id": ids[0], "last_synthetic_id": ids[-1],
                      "physics_count": sum(1 for row in group if row["route"] == "physics"),
                      "gpat_count": sum(1 for row in group if row["route"] == "gpat"),
                      "live_casia_fasd_count": sum(1 for row in group if row["live_target_dataset"] == "casia_fasd"),
                      "live_msu_mfsd_count": sum(1 for row in group if row["live_target_dataset"] == "msu_mfsd")})
    return index


def write_shards_index(bank_root: Path, index: list[dict[str, Any]]) -> Path:
    table = pa.Table.from_pydict({name: [row[name] for row in index] for name, _ in _INDEX_FIELDS},
                                 schema=pa.schema(_INDEX_FIELDS))
    target = Path(bank_root) / "shards_index.parquet"
    pq.write_table(table, target, compression="none", version="2.6", write_statistics=False)
    return target


def load_shards_index(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(Path(path)).to_pydict()
    return [{name: table[name][index] for name, _ in _INDEX_FIELDS} for index in range(len(table["shard_name"]))]


def index_digest(index: list[dict[str, Any]]) -> str:
    import json
    canonical = json.dumps([{name: row[name] for name, _ in _INDEX_FIELDS} for row in index],
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_shard_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(Path(path), mode="r:") as archive:
        return {member.name: (archive.extractfile(member).read() if member.isfile() else b"")
                for member in archive.getmembers()}


def shard_member_metadata(path: Path) -> list[dict[str, Any]]:
    with tarfile.open(Path(path), mode="r:") as archive:
        return [{"name": member.name, "size": member.size, "mtime": member.mtime, "mode": member.mode,
                 "uid": member.uid, "gid": member.gid, "uname": member.uname, "gname": member.gname,
                 "is_file": member.isfile()} for member in archive.getmembers()]


def validate_shards(bank_root: Path, index: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute every shard hash and prove loose/shard parity on the audited
    members: a shard byte stream must equal the loose files it was built from."""
    root = Path(bank_root)
    errors: list[str] = []
    seen: list[str] = []
    for row in index:
        path = root / "shards" / row["shard_name"]
        if not path.is_file(): errors.append(f"missing shard {row['shard_name']}"); continue
        payload = path.read_bytes()
        if len(payload) != int(row["byte_size"]): errors.append(f"{row['shard_name']}: byte size mismatch")
        if hashlib.sha256(payload).hexdigest() != row["sha256"]: errors.append(f"{row['shard_name']}: sha mismatch")
        for member in shard_member_metadata(path):
            if member["mtime"] != TAR_MTIME or member["uid"] != TAR_UID or member["gid"] != TAR_GID \
                    or member["uname"] != TAR_UNAME or member["gname"] != TAR_GNAME or member["mode"] != TAR_MODE:
                errors.append(f"{row['shard_name']}: member {member['name']} has non-normalized metadata"); break
        members = read_shard_members(path)
        ids = sorted({name.split(".")[0] for name in members})
        seen.extend(ids)
        if len(ids) != int(row["row_count"]): errors.append(f"{row['shard_name']}: row count mismatch")
        for synthetic_id in ids:
            for name, source in _member_sources(root, synthetic_id):
                if name not in members: errors.append(f"{row['shard_name']}: missing member {name}"); continue
                if not source.is_file(): errors.append(f"loose file for {name} is missing"); continue
                if members[name] != source.read_bytes():
                    errors.append(f"{row['shard_name']}: member {name} differs from the loose file")
    expected = sorted(row["synthetic_id"] for row in accepted)
    return {"shards": len(index), "members_checked": len(seen), "parity_ids": len(set(seen)),
            "covers_every_accepted_row": sorted(set(seen)) == expected,
            "errors": errors[:50], "error_count": len(errors), "passed": not errors and sorted(set(seen)) == expected}
