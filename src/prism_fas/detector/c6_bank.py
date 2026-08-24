"""The detector's read-only view of one arm's frozen C6 matched bank.

Version B's detector opened exactly one synthetic bank: the frozen M8 v3 export,
through `detector.synthetic_bank.SyntheticBankReader`. Version C's detector
trains on something else — the 1024 candidates C6 selected for one arm, whose
bytes live in the C5 candidate tree and whose membership lives in that arm's
`C6_BANK_LOCK_<ARM>.json`. Those are different artifacts in different shapes,
and there is no honest way to make one pretend to be the other.

So this module supplies a second reader rather than a translation. It presents
exactly the surface `M9TrainingDataset` and `region_cache` consume — `rows`,
`sample`, `bank_id`, `identity`, `rows_identity`, `primary_artifact_family`,
`lock` — and it is fail-closed in the same three ways the M8 reader is:

* **Membership comes from the lock, never from the directory.** The bank is the
  1024 candidate ids C6 selected. A candidate sitting in the C5 tree that C6 did
  not select is not in this bank, and a selected candidate whose directory is
  missing is an error rather than a shorter bank.
* **Bytes are verified against the record that wrote them.** Every payload is
  re-hashed and compared to the `payload_sha256` C5 recorded, so a corrupted or
  re-rendered candidate fails here rather than becoming a training sample.
* **Only `GENERATED` records are readable.** A retained C5 semantic failure has
  no payload; it is negative provenance, and asking for it raises.

`q` arrives from the C6 lock and is carried as a §11.2 TRAINING WEIGHT. Nothing
here derives a label from it, and nothing here consults it to decide membership —
C6 already decided that, without it.

Reads the C5 candidate tree, the arm's C3 recipe bank and the C6 locks. Opens no
source package, no target and no network.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from prism_fas.detector.synthetic_bank import SyntheticSample

SCHEMA_VERSION = "prism-c6-matched-bank-reader-v1"

#: The backend name this reader reports. Deliberately not "loose": a downstream
#: artifact that says `backend: loose` is claiming an M8 v3 export.
BACKEND = "c6_matched_bank"

#: Bank ids are namespaced so a C6 arm bank can never satisfy a pin written for
#: the frozen M8 v3 bank, or the reverse.
BANK_ID_PREFIX = "prism_c6_matched_bank"


class C6BankError(RuntimeError):
    """The arm's matched bank cannot be assembled from what is on disk."""

    reason_code = "C6_MATCHED_BANK_UNREADABLE"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def recipe_index(recipes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The arm's C3 recipes keyed by id, reduced to what a sample carries.

    `artifact_types` feeds `artifact_family_risk` grouping and `regions` is
    provenance; both are properties of the recipe, so they are read from the
    frozen recipe rather than re-derived from the rendered pixels.
    """
    index: dict[str, dict[str, Any]] = {}
    for recipe in recipes:
        identifier = str(recipe.get("recipe_id") or "")
        if not identifier:
            continue
        artifacts = [str(item.get("name") or "") for item in recipe.get("artifacts") or ()]
        index[identifier] = {
            "artifact_types": tuple(sorted(name for name in artifacts if name)),
            "regions": tuple(str(name) for name in recipe.get("regions") or ()),
            "recipe_hash": hashlib.sha256(_canonical(recipe).encode("utf-8")).hexdigest(),
        }
    return index


@dataclass(frozen=True)
class C6MatchedBankReader:
    """A verified, read-only handle on one arm's frozen C6 matched bank.

    Construction performs every membership and identity check, so no caller can
    reach a sample through an unverified path — the same property that makes
    `SyntheticBankReader.open` the only way to a frozen M8 sample.
    """

    root: Path
    arm: str
    lock: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    backend: str = BACKEND
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # --- construction ---------------------------------------------------------

    @classmethod
    def open(cls, *, candidates_root: Path, arm: str, bank_lock: Mapping[str, Any],
             recipes: Sequence[Mapping[str, Any]], package_identity: str,
             recipe_bank_identity: str,
             expected_selected_set_sha256: str | None = None) -> "C6MatchedBankReader":
        """Assemble the arm's bank from its lock and the C5 candidate tree.

        `package_identity` is passed in rather than read from a candidate record
        because `M9TrainingDataset` compares it to the source package it opened;
        taking it from the caller means the comparison is between the package
        this run resolved and the package C5 rendered against, which is the
        comparison that matters. It is still checked row by row below.
        """
        from prism_fas.synthesis import c5_raw_generation as raw

        root = Path(candidates_root)
        selected = list(bank_lock.get("selected") or ())
        if not selected:
            raise C6BankError(
                f"the {arm} C6 bank lock serializes no selected rows; a matched bank "
                "is its selected set and cannot be recovered from the directory")
        if expected_selected_set_sha256 and str(
                bank_lock.get("selected_set_sha256")) != expected_selected_set_sha256:
            raise C6BankError(
                f"the {arm} bank lock records selected set "
                f"{bank_lock.get('selected_set_sha256')!r} but "
                f"{expected_selected_set_sha256!r} was expected")

        recipes_by_id = recipe_index(recipes)
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for entry in selected:
            candidate_id = str(entry.get("candidate_id") or "")
            directory = raw.candidate_dir(root, arm, candidate_id)
            record = raw.read_record(directory / raw.RECORD_NAME)
            if record is None:
                missing.append(candidate_id)
                continue
            rows.append(_row(arm=arm, entry=entry, record=record, directory=directory,
                             recipes_by_id=recipes_by_id,
                             recipe_bank_identity=recipe_bank_identity,
                             threshold_hash=str(
                                 bank_lock.get("quality_threshold_identity") or "")))
        if missing:
            raise C6BankError(
                f"{len(missing)} of the {len(selected)} candidates the {arm} C6 bank "
                f"selected have no C5 record under {root.as_posix()}, starting at "
                f"{missing[:3]}. A matched bank is never silently shortened")

        disagreeing = sorted({row["package_identity"] for row in rows} - {package_identity})
        if disagreeing:
            raise C6BankError(
                f"the {arm} bank holds candidates rendered against source package(s) "
                f"{disagreeing[:2]} but this run resolved {package_identity!r}")

        rows.sort(key=lambda row: row["synthetic_id"])
        lock = _lock_payload(arm=arm, bank_lock=bank_lock, rows=rows,
                             package_identity=package_identity,
                             recipe_bank_identity=recipe_bank_identity,
                             candidates_root=root)
        return cls(root=root, arm=arm, lock=lock, rows=tuple(rows))

    # --- identity -------------------------------------------------------------

    @property
    def bank_id(self) -> str:
        return str(self.lock["bank_id"])

    @property
    def identity(self) -> str:
        return str(self.lock["bank_content_identity_sha256"])

    @property
    def synthetic_ids(self) -> tuple[str, ...]:
        return tuple(row["synthetic_id"] for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def rows_identity(self) -> str:
        """Digest over the rows this reader will actually yield.

        Same fields the M8 reader digests, so the two identities are comparable
        quantities even though the banks are not interchangeable.
        """
        payload = [{"synthetic_id": row["synthetic_id"], "route": row["route"],
                    "recipe_id": row["recipe_id"], "q": format(float(row["q"]), ".12g"),
                    "image_sha256": row["image_sha256"], "mask_sha256": row["mask_sha256"],
                    "artifact_map_sha256": row["artifact_map_sha256"]}
                   for row in self.rows]
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def index_of(self, synthetic_id: str) -> int:
        try:
            return self.synthetic_ids.index(synthetic_id)
        except ValueError:
            raise KeyError(
                f"{synthetic_id!r} is not a member of the {self.arm} C6 matched "
                "bank") from None

    def row(self, position: int) -> dict[str, Any]:
        return dict(self.rows[int(position)])

    def primary_artifact_family(self, position: int) -> str:
        types = [value for value in
                 str(self.rows[int(position)]["artifact_types"]).split("|") if value]
        return sorted(types)[0] if types else ""

    def summary(self) -> dict[str, Any]:
        routes: dict[str, int] = {}
        domains: dict[str, int] = {}
        for row in self.rows:
            routes[row["route"]] = routes.get(row["route"], 0) + 1
            domains[row["live_target_dataset"]] = domains.get(row["live_target_dataset"], 0) + 1
        return {"schema_version": SCHEMA_VERSION, "bank_id": self.bank_id,
                "bank_content_identity_sha256": self.identity,
                "status": str(self.lock["status"]), "backend": self.backend,
                "arm": self.arm, "accepted": len(self.rows),
                "route_counts": dict(sorted(routes.items())),
                "live_target_datasets": dict(sorted(domains.items())),
                "package_identity": str(self.lock["package_identity"]),
                "recipe_bank_identity": str(self.lock["recipe_bank_identity"]),
                "c6_selected_set_sha256": str(self.lock["c6_selected_set_sha256"]),
                "c6_selector_identity_sha256": str(self.lock["c6_selector_identity_sha256"]),
                "rows_identity_sha256": self.rows_identity()}

    # --- payload --------------------------------------------------------------

    def _payload(self, row: Mapping[str, Any]) -> tuple[bytes, bytes, bytes]:
        """The three payload files, each re-hashed against the C5 record.

        A candidate whose bytes moved after C5 recorded them is refused. C5's own
        `reuse_decision` treats a changed payload as a rebuild trigger; by the
        time C6 has selected it, a changed payload means the bank C6 froze is not
        the bank on disk, and training on it would silently substitute one.
        """
        parts: list[bytes] = []
        for path_key, digest_key in (("image_relative_path", "image_sha256"),
                                     ("mask_relative_path", "mask_sha256"),
                                     ("artifact_map_relative_path", "artifact_map_sha256")):
            path = self.root / str(row[path_key])
            if not path.is_file():
                raise C6BankError(
                    f"{row['synthetic_id']}: {path_key} is missing at {path.as_posix()}")
            payload = path.read_bytes()
            if _sha256_bytes(payload) != row[digest_key]:
                raise C6BankError(
                    f"{row['synthetic_id']}: {path_key} does not match the SHA-256 "
                    "the C5 candidate record wrote for it")
            parts.append(payload)
        return parts[0], parts[1], parts[2]

    def sample(self, position: int) -> SyntheticSample:
        """One validated sample, in exactly the shape the M8 reader yields."""
        from prism_fas.synthesis.synthetic_bank import decode_npz, decode_png, from_uint8

        row = self.rows[int(position)]
        image_png, mask_png, artifact_npz = self._payload(row)
        image = from_uint8(decode_png(image_png))
        mask_image = decode_png(mask_png)
        if mask_image.ndim != 2 or set(np.unique(mask_image).tolist()) - {0, 255}:
            raise C6BankError(f"{row['synthetic_id']}: mask is not a 0/255 map")
        artifact = np.asarray(decode_npz(artifact_npz), dtype=np.float32)
        return SyntheticSample(
            synthetic_id=str(row["synthetic_id"]), image=image,
            exact_mask=(mask_image == 255), artifact_map=artifact,
            route=str(row["route"]), recipe_id=str(row["recipe_id"]),
            recipe_hash=str(row["recipe_hash"]), graph_hash=str(row["graph_hash"]),
            artifact_types=tuple(value for value in str(row["artifact_types"]).split("|")
                                 if value),
            regions=tuple(value for value in str(row["regions"]).split("|") if value),
            quality_weight=float(row["q"]),
            live_target_sample_id=str(row["live_target_sample_id"]),
            live_target_dataset=str(row["live_target_dataset"]),
            spoof_source_dataset=str(row["spoof_source_dataset"]),
            domain_relation=str(row["domain_relation"]),
            exact_mask_pixels=int(row["exact_mask_pixels"]),
            threshold_hash=str(row["threshold_hash"]),
            package_identity=str(row["package_identity"]),
            recipe_bank_identity=str(row["recipe_bank_identity"])).validate()

    def __getitem__(self, position: int) -> SyntheticSample:
        return self.sample(position)

    def __iter__(self) -> Iterator[SyntheticSample]:
        for position in range(len(self.rows)):
            yield self.sample(position)


def _row(*, arm: str, entry: Mapping[str, Any], record: Mapping[str, Any],
         directory: Path, recipes_by_id: Mapping[str, Mapping[str, Any]],
         recipe_bank_identity: str, threshold_hash: str) -> dict[str, Any]:
    """One bank row, from the C6 selection and the C5 record together.

    Membership, `q` and the source domain come from C6; the bytes, the generation
    identity and the render trace come from C5. Neither is asked for something it
    does not own.
    """
    from prism_fas.synthesis import c5_raw_generation as raw

    candidate_id = str(entry.get("candidate_id") or "")
    status = str(record.get("status"))
    if status != raw.GENERATED:
        raise C6BankError(
            f"{candidate_id}: the C5 record is {status!r}, not {raw.GENERATED!r}. A "
            "retained generation failure has no payload and is negative provenance, "
            "never a training sample")

    identity = dict(record.get("generation_identity") or {})
    payloads = dict(record.get("payload_sha256") or {})
    for name in raw.PAYLOAD_NAMES:
        if not payloads.get(name):
            raise C6BankError(f"{candidate_id}: the C5 record has no SHA-256 for {name}")

    trace = dict(record.get("trace") or {})
    recipe_id = str(identity.get("recipe_id") or entry.get("recipe_id") or "")
    recipe = dict(recipes_by_id.get(recipe_id) or {})
    if not recipe:
        raise C6BankError(
            f"{candidate_id}: recipe {recipe_id!r} is not in the {arm} C3 bank, so its "
            "artifact family and regions cannot be resolved. The bank and the "
            "candidates were built from different recipe sets")

    from prism_fas.synthesis.c5_source_pair_plan import domain_relation_for_slot

    relative = directory.relative_to(directory.parents[1])
    domain = str(entry.get("source_domain") or "")
    # Both are provenance the detector carries but never optimizes on. The slot
    # is in the generation identity, so the relation is recoverable exactly; the
    # spoof source's DATASET is not recorded by C5 at all and is therefore named
    # as unrecorded rather than defaulted to the live domain, which would assert
    # a same-domain pairing the record does not support.
    relation = domain_relation_for_slot(int(identity.get("slot", -1))) or ""
    spoof_source = identity.get("spoof_source_sample_id")
    return {
        # `synthetic_id` IS the C5 candidate id. Minting a second identifier would
        # give the same sample two names across C5, C6 and the detector.
        "synthetic_id": candidate_id,
        "arm": arm,
        "route": str(identity.get("route") or entry.get("route") or ""),
        "recipe_id": recipe_id,
        "recipe_ordinal": int(identity.get("recipe_ordinal", entry.get("recipe_ordinal", -1))),
        "recipe_hash": str(recipe["recipe_hash"]),
        "graph_hash": str(record.get("generation_identity_sha256") or ""),
        "artifact_types": "|".join(recipe["artifact_types"]),
        "regions": "|".join(recipe["regions"]),
        "live_target_sample_id": str(identity.get("live_target_sample_id")
                                     or entry.get("live_target_sample_id") or ""),
        "live_target_dataset": domain,
        "spoof_source_sample_id": str(spoof_source or ""),
        "spoof_source_dataset": ("" if spoof_source is None else
                                 "NOT_RECORDED_BY_C5_CANDIDATE_RECORD"),
        "domain_relation": relation,
        "image_relative_path": (relative / raw.IMAGE_NAME).as_posix(),
        "mask_relative_path": (relative / raw.MASK_NAME).as_posix(),
        "artifact_map_relative_path": (relative / raw.ARTIFACT_MAP_NAME).as_posix(),
        "image_sha256": str(payloads[raw.IMAGE_NAME]),
        "mask_sha256": str(payloads[raw.MASK_NAME]),
        "artifact_map_sha256": str(payloads[raw.ARTIFACT_MAP_NAME]),
        # A §11.2 training weight, carried from C6's decision record. It played no
        # part in selection and never becomes a label.
        "q": float(entry.get("q") if entry.get("q") is not None else 1.0),
        "exact_mask_pixels": int(trace.get("exact_mask_pixels", 0)),
        "threshold_hash": threshold_hash,
        "package_identity": str(identity.get("package_identity") or ""),
        "recipe_bank_identity": recipe_bank_identity,
        "c3_bank_identity": str(identity.get("recipe_bank_identity") or ""),
        "generation_identity_sha256": str(record.get("generation_identity_sha256") or ""),
        "base_position": int(entry.get("base_position", identity.get("position", -1))),
    }


def _lock_payload(*, arm: str, bank_lock: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                  package_identity: str, recipe_bank_identity: str,
                  candidates_root: Path) -> dict[str, Any]:
    """The reader's own lock: what this bank is, and which C6 decision made it."""
    ordered = sorted(rows, key=lambda row: row["synthetic_id"])
    material = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "c6_selected_set_sha256": str(bank_lock.get("selected_set_sha256") or ""),
        "c6_selector_identity_sha256": str(bank_lock.get("selector_identity_sha256") or ""),
        "quality_profile": str(bank_lock.get("quality_profile") or ""),
        "quality_threshold_identity": str(bank_lock.get("quality_threshold_identity") or ""),
        "package_identity": package_identity,
        "recipe_bank_identity": recipe_bank_identity,
        "rows": [{"synthetic_id": row["synthetic_id"], "route": row["route"],
                  "recipe_id": row["recipe_id"],
                  "q": format(float(row["q"]), ".12g"),
                  "image_sha256": row["image_sha256"],
                  "mask_sha256": row["mask_sha256"],
                  "artifact_map_sha256": row["artifact_map_sha256"]} for row in ordered],
    }
    content = hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()
    return {
        **{key: value for key, value in material.items() if key != "rows"},
        "bank_id": f"{BANK_ID_PREFIX}_{arm.lower()}_{content[:12]}",
        "bank_content_identity_sha256": content,
        "status": "validated",
        "accepted_count": len(ordered),
        "candidates_root": candidates_root.as_posix(),
        "q_used_for_selection": False,
        "q_purpose": "§11.2 synthetic sample-quality TRAINING WEIGHT only",
        "target_access": 0,
    }


def open_arm_bank(repo: Path, *, arm: str, evidence: Any, candidates_root: Path,
                  package_identity: str,
                  recipe_bank_identity: str) -> C6MatchedBankReader:
    """Open one arm's matched bank from verified C6 evidence.

    `evidence` is an `c6_evidence.ArmBankEvidence`, so the caller cannot reach
    this function without having verified the C6 closure first — the bank lock is
    re-read here from the path that evidence names rather than from a directory
    scan.
    """
    from prism_fas.synthesis.c5_arm_plan import load_arm_bank

    lock_path = Path(repo) / evidence.lock_path
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    bank = load_arm_bank(Path(repo), arm)
    return C6MatchedBankReader.open(
        candidates_root=candidates_root, arm=arm, bank_lock=payload,
        recipes=bank["recipes"], package_identity=package_identity,
        recipe_bank_identity=recipe_bank_identity,
        expected_selected_set_sha256=evidence.selected_set_sha256)


__all__ = ["SCHEMA_VERSION", "BACKEND", "BANK_ID_PREFIX", "C6BankError",
           "C6MatchedBankReader", "recipe_index", "open_arm_bank"]
