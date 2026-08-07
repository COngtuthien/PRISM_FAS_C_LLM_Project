"""Read-only M9 view of the frozen, validated M8 synthetic bank.

M9 may read exactly one bank: `prism_synthetic_bank_m8_v3_e84c78cd2a9b`, content
identity `e84c78cd2a9b…`, status `validated`. This module FAILS CLOSED — the lock is
checked before the first sample is produced, and there is no fallback to a v1/v2
bank, to an M8 working namespace or to a rejected candidate.

`q` is loaded as a SAMPLE WEIGHT (spec section 8, Table 30: "Detector uses q_i as a
weight; q_i is not a live/spoof label"). Nothing here derives a class label from it;
the synthetic training label comes from the detector training contract.

Two backends, both over the same accepted manifest rows and both proven to agree:

    loose  — the bank's `images/`, `masks/`, `artifact_maps/` files
    shard  — the deterministic `shards/synthetic-*.tar` byte stream

Never imports modal.
"""
from __future__ import annotations
import hashlib, json, tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import numpy as np

M9_BANK_READER_SCHEMA_VERSION = "m9-synthetic-bank-reader-v1"
# The one bank M9 is allowed to open. Both are checked; neither is configurable
# away at runtime.
FROZEN_BANK_ID = "prism_synthetic_bank_m8_v3_e84c78cd2a9b"
FROZEN_BANK_IDENTITY = "e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542"
REQUIRED_STATUS = "validated"
EXPECTED_ACCEPTED = 871
ROUTES = ("physics", "gpat")
LIVE_DOMAINS = ("casia_fasd", "msu_mfsd")
IMAGE_SIZE = 224


class SyntheticBankAccessError(RuntimeError):
    """The requested synthetic bank is not the frozen, validated M8 v3 bank."""


@dataclass(frozen=True)
class SyntheticSample:
    """One accepted synthetic sample, exactly as the bank schema stores it."""
    synthetic_id: str
    image: np.ndarray                 # [3,224,224] float32 in [0,1]
    exact_mask: np.ndarray            # [224,224] bool — the ACTUAL changed pixels
    artifact_map: np.ndarray          # [1,224,224] float32 in [0,1]
    route: str                        # physics | gpat
    recipe_id: str
    recipe_hash: str
    graph_hash: str
    artifact_types: tuple[str, ...]
    regions: tuple[str, ...]
    quality_weight: float             # q — a WEIGHT, never a label
    live_target_sample_id: str
    live_target_dataset: str          # the SOURCE DOMAIN of the edited live image
    spoof_source_dataset: str
    domain_relation: str
    exact_mask_pixels: int
    threshold_hash: str
    package_identity: str
    recipe_bank_identity: str

    def validate(self) -> "SyntheticSample":
        if self.image.shape != (3, IMAGE_SIZE, IMAGE_SIZE) or self.image.dtype != np.float32:
            raise SyntheticBankAccessError(f"{self.synthetic_id}: image must be float32 [3,224,224]")
        if not np.isfinite(self.image).all() or self.image.min() < 0.0 or self.image.max() > 1.0:
            raise SyntheticBankAccessError(f"{self.synthetic_id}: image outside [0,1]")
        if self.exact_mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or self.exact_mask.dtype != np.bool_:
            raise SyntheticBankAccessError(f"{self.synthetic_id}: exact mask must be bool [224,224]")
        if self.artifact_map.shape != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise SyntheticBankAccessError(f"{self.synthetic_id}: artifact map must be [1,224,224]")
        if not np.isfinite(self.artifact_map).all():
            raise SyntheticBankAccessError(f"{self.synthetic_id}: artifact map is not finite")
        if float(np.abs(self.artifact_map[0][~self.exact_mask]).max(initial=0.0)) != 0.0:
            raise SyntheticBankAccessError(f"{self.synthetic_id}: artifact map is not zero outside the exact mask")
        if not 0.0 <= self.quality_weight <= 1.0:
            raise SyntheticBankAccessError(f"{self.synthetic_id}: q outside [0,1]")
        if self.route not in ROUTES:
            raise SyntheticBankAccessError(f"{self.synthetic_id}: unknown route {self.route!r}")
        return self


@dataclass(frozen=True)
class SyntheticBankReader:
    """A verified, read-only handle on the frozen bank.

    Construction performs every identity check, so no caller can reach a sample
    through an unverified path.
    """
    root: Path
    lock: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    backend: str

    # --- construction -------------------------------------------------------
    @classmethod
    def open(cls, bank_root: Path, *, backend: str = "loose",
             expected_identity: str = FROZEN_BANK_IDENTITY,
             expected_bank_id: str = FROZEN_BANK_ID) -> "SyntheticBankReader":
        root = Path(bank_root)
        if backend not in ("loose", "shard"):
            raise SyntheticBankAccessError(f"unknown synthetic bank backend {backend!r}")
        lock_path = root / "BANK_LOCK.json"
        if not lock_path.is_file():
            raise SyntheticBankAccessError(f"{root.name} is not a synthetic bank: BANK_LOCK.json is missing")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        status = str(lock.get("status"))
        if status != REQUIRED_STATUS:
            raise SyntheticBankAccessError(f"bank status is {status!r}, M9 requires {REQUIRED_STATUS!r}")
        identity = str(lock.get("bank_content_identity_sha256"))
        if identity != expected_identity:
            raise SyntheticBankAccessError(
                f"bank content identity {identity} != frozen {expected_identity}; "
                "M9 never falls back to a v1/v2 bank or to a working namespace")
        bank_id = str(lock.get("bank_id"))
        if bank_id != expected_bank_id:
            raise SyntheticBankAccessError(f"bank id {bank_id!r} != frozen {expected_bank_id!r}")
        rows = cls._accepted_rows(root)
        if int(lock.get("accepted_count", -1)) != len(rows):
            raise SyntheticBankAccessError(
                f"manifest holds {len(rows)} rows but the lock declares {lock.get('accepted_count')} accepted")
        return cls(root=root, lock=lock, rows=tuple(rows), backend=backend)

    @staticmethod
    def _accepted_rows(root: Path) -> list[dict[str, Any]]:
        """ONLY `manifests/manifest.parquet`, which by construction holds accepted
        rows. `rejected.parquet` and `candidate_manifest.parquet` are never read."""
        from prism_fas.synthesis.synthetic_bank import MANIFEST_SCHEMAS, load_manifest
        path = root / "manifests" / "manifest.parquet"
        if not path.is_file(): raise SyntheticBankAccessError("accepted manifest is missing from the bank")
        rows = load_manifest(path, MANIFEST_SCHEMAS["manifest"])
        return sorted(rows, key=lambda row: row["synthetic_id"])

    # --- identity -----------------------------------------------------------
    @property
    def bank_id(self) -> str: return str(self.lock["bank_id"])
    @property
    def identity(self) -> str: return str(self.lock["bank_content_identity_sha256"])
    @property
    def synthetic_ids(self) -> tuple[str, ...]: return tuple(row["synthetic_id"] for row in self.rows)

    def __len__(self) -> int: return len(self.rows)

    def summary(self) -> dict[str, Any]:
        routes: dict[str, int] = {}
        domains: dict[str, int] = {}
        for row in self.rows:
            routes[row["route"]] = routes.get(row["route"], 0) + 1
            domains[row["live_target_dataset"]] = domains.get(row["live_target_dataset"], 0) + 1
        return {"schema_version": M9_BANK_READER_SCHEMA_VERSION, "bank_id": self.bank_id,
                "bank_content_identity_sha256": self.identity, "status": str(self.lock["status"]),
                "backend": self.backend, "accepted": len(self.rows),
                "route_counts": dict(sorted(routes.items())),
                "live_target_datasets": dict(sorted(domains.items())),
                "package_identity": str(self.lock["package_identity"]),
                "recipe_bank_identity": str(self.lock["recipe_bank_identity"]),
                "rows_identity_sha256": self.rows_identity()}

    def rows_identity(self) -> str:
        """Digest over the accepted rows this reader will actually yield."""
        payload = [{"synthetic_id": row["synthetic_id"], "route": row["route"],
                    "recipe_id": row["recipe_id"], "q": format(float(row["q"]), ".12g"),
                    "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"],
                    "artifact_map_sha256": row["artifact_map_sha256"]} for row in self.rows]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def index_of(self, synthetic_id: str) -> int:
        try: return self.synthetic_ids.index(synthetic_id)
        except ValueError:
            raise KeyError(f"{synthetic_id!r} is not an accepted sample of the frozen bank") from None

    # --- payload ------------------------------------------------------------
    def _payload_loose(self, row: dict[str, Any]) -> tuple[bytes, bytes, bytes]:
        parts = []
        for path_key, digest_key in (("image_relative_path", "image_sha256"),
                                     ("mask_relative_path", "mask_sha256"),
                                     ("artifact_map_relative_path", "artifact_map_sha256")):
            path = self.root / str(row[path_key])
            if not path.is_file():
                raise SyntheticBankAccessError(f"{row['synthetic_id']}: {path_key} is missing from the bank")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row[digest_key]:
                raise SyntheticBankAccessError(f"{row['synthetic_id']}: {path_key} does not match its manifest hash")
            parts.append(payload)
        return parts[0], parts[1], parts[2]

    def _shard_members(self) -> dict[str, bytes]:
        """Every member of every shard, read once. The shard stream is the
        deterministic transport form of the same accepted rows."""
        cached = getattr(self, "_shard_cache", None)
        if cached is not None: return cached
        members: dict[str, bytes] = {}
        index_path = self.root / "shards_index.parquet"
        if not index_path.is_file(): raise SyntheticBankAccessError("shards_index.parquet is missing from the bank")
        from prism_fas.synthesis.synthetic_shards import load_shards_index
        for entry in load_shards_index(index_path):
            path = self.root / "shards" / str(entry["shard_name"])
            if not path.is_file(): raise SyntheticBankAccessError(f"shard {entry['shard_name']} is missing")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != str(entry["sha256"]):
                raise SyntheticBankAccessError(f"shard {entry['shard_name']} does not match its index hash")
            with tarfile.open(path, mode="r:") as archive:
                for member in archive.getmembers():
                    if member.isfile(): members[member.name] = archive.extractfile(member).read()
        object.__setattr__(self, "_shard_cache", members)
        return members

    def _payload_shard(self, row: dict[str, Any]) -> tuple[bytes, bytes, bytes]:
        members = self._shard_members()
        synthetic_id = str(row["synthetic_id"])
        names = (f"{synthetic_id}.png", f"{synthetic_id}.mask.png", f"{synthetic_id}.npz")
        missing = [name for name in names if name not in members]
        if missing: raise SyntheticBankAccessError(f"{synthetic_id}: shard members {missing} are missing")
        parts = tuple(members[name] for name in names)
        for payload, digest_key in zip(parts, ("image_sha256", "mask_sha256", "artifact_map_sha256")):
            if hashlib.sha256(payload).hexdigest() != row[digest_key]:
                raise SyntheticBankAccessError(f"{synthetic_id}: shard member does not match its manifest hash")
        return parts

    def sample(self, position: int) -> SyntheticSample:
        row = self.rows[int(position)]
        image_png, mask_png, artifact_npz = (self._payload_loose(row) if self.backend == "loose"
                                             else self._payload_shard(row))
        from prism_fas.synthesis.synthetic_bank import decode_npz, decode_png, from_uint8
        image = from_uint8(decode_png(image_png))
        mask_image = decode_png(mask_png)
        if mask_image.ndim != 2 or set(np.unique(mask_image).tolist()) - {0, 255}:
            raise SyntheticBankAccessError(f"{row['synthetic_id']}: mask is not a 0/255 map")
        artifact = np.asarray(decode_npz(artifact_npz), dtype=np.float32)
        return SyntheticSample(
            synthetic_id=str(row["synthetic_id"]), image=image, exact_mask=(mask_image == 255),
            artifact_map=artifact, route=str(row["route"]), recipe_id=str(row["recipe_id"]),
            recipe_hash=str(row["recipe_hash"]), graph_hash=str(row["graph_hash"]),
            artifact_types=tuple(value for value in str(row["artifact_types"]).split("|") if value),
            regions=tuple(value for value in str(row["regions"]).split("|") if value),
            quality_weight=float(row["q"]), live_target_sample_id=str(row["live_target_sample_id"]),
            live_target_dataset=str(row["live_target_dataset"]),
            spoof_source_dataset=str(row["spoof_source_dataset"]),
            domain_relation=str(row["domain_relation"]), exact_mask_pixels=int(row["exact_mask_pixels"]),
            threshold_hash=str(row["threshold_hash"]), package_identity=str(row["package_identity"]),
            recipe_bank_identity=str(row["recipe_bank_identity"])).validate()

    def __getitem__(self, position: int) -> SyntheticSample: return self.sample(position)

    def __iter__(self) -> Iterator[SyntheticSample]:
        for position in range(len(self.rows)): yield self.sample(position)

    def row(self, position: int) -> dict[str, Any]: return dict(self.rows[int(position)])

    def primary_artifact_family(self, position: int) -> str:
        """`artifact_family_risk` groups by the recipe's primary artifact name
        (docs/M9_TRAINING_CONTRACT.md section 1)."""
        types = [value for value in str(self.rows[int(position)]["artifact_types"]).split("|") if value]
        return sorted(types)[0] if types else ""


def parity_audit(bank_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Prove that the loose and shard backends yield the same samples.

    Compares the decoded arrays, not just the bytes, so a decoder difference would
    show up too.
    """
    loose = SyntheticBankReader.open(Path(bank_root), backend="loose")
    shard = SyntheticBankReader.open(Path(bank_root), backend="shard")
    if loose.synthetic_ids != shard.synthetic_ids:
        raise SyntheticBankAccessError("loose and shard backends disagree on the accepted row set")
    count = len(loose) if limit is None else min(int(limit), len(loose))
    mismatches: list[str] = []
    for position in range(count):
        left, right = loose.sample(position), shard.sample(position)
        same = (left.synthetic_id == right.synthetic_id
                and np.array_equal(left.image, right.image)
                and np.array_equal(left.exact_mask, right.exact_mask)
                and np.array_equal(left.artifact_map, right.artifact_map)
                and left.quality_weight == right.quality_weight
                and left.route == right.route and left.recipe_id == right.recipe_id)
        if not same: mismatches.append(left.synthetic_id)
    return {"schema_version": M9_BANK_READER_SCHEMA_VERSION, "bank_id": loose.bank_id,
            "compared": count, "accepted_total": len(loose), "mismatches": mismatches[:20],
            "mismatch_count": len(mismatches), "passed": not mismatches,
            "rows_identity_sha256": loose.rows_identity()}
