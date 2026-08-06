from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq
from prism_fas.recipes.bank import load_bank
from prism_fas.utils.core import atomic_json_write

PAIR_PLAN_SCHEMA_VERSION = "m8-pair-plan-v1"
PAIR_PLAN_SEED = 20260806
SOURCE_SPLIT = "source_train"
ALLOWED_DATASETS = ("casia_fasd", "msu_mfsd")
TRAIN_FRACTION = 0.8
PAIRS_PER_LIVE = 4              # 2 same-domain + 2 cross-domain spoof sources
SAME_DOMAIN_PER_LIVE = 2
CROSS_DOMAIN_PER_LIVE = 2
EXPECTED_TRAIN_PAIRS = 896
EXPECTED_VALIDATION_PAIRS = 224


class PairPlanError(ValueError):
    """The deterministic pair plan cannot be built as declared."""


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _bucket(value: str, modulus: int = 1_000_000) -> int:
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "big") % modulus


@dataclass(frozen=True)
class SourceRow:
    sample_id: str
    dataset: str
    source_record_id: str
    subject_id: str | None
    label: str
    project_split: str

    def as_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "dataset": self.dataset, "source_record_id": self.source_record_id,
                "subject_id": self.subject_id, "label": self.label, "project_split": self.project_split}


def load_source_train_rows(package_root: Path) -> list[SourceRow]:
    """Read `manifests/source_train.parquet` only.

    `source_dev` and `target_test` manifests are never opened by M8.
    """
    manifest = Path(package_root) / "manifests" / f"{SOURCE_SPLIT}.parquet"
    if not manifest.is_file(): raise PairPlanError(f"missing {SOURCE_SPLIT} manifest")
    table = pq.read_table(manifest).to_pydict()
    rows: list[SourceRow] = []
    for index in range(len(table["sample_id"])):
        split = table["project_split"][index]
        dataset = table["dataset"][index]
        if split != SOURCE_SPLIT: raise PairPlanError(f"manifest row {index} has project_split {split!r}")
        if dataset not in ALLOWED_DATASETS: raise PairPlanError(f"manifest row {index} has dataset {dataset!r}")
        subject = table["subject_id"][index]
        rows.append(SourceRow(sample_id=table["sample_id"][index], dataset=dataset,
                              source_record_id=table["source_record_id"][index],
                              subject_id=str(subject) if subject not in (None, "") else None,
                              label=table["label_live_spoof"][index], project_split=split))
    return sorted(rows, key=lambda row: row.sample_id)


def partition_records(rows: list[SourceRow], label: str) -> dict[str, str]:
    """Deterministic 80/20 record partition, stratified by dataset.

    A `source_record_id` belongs to exactly one partition for a given role, so a
    record can never appear in both GPAT train and GPAT validation.
    """
    assignment: dict[str, str] = {}
    for dataset in ALLOWED_DATASETS:
        records = sorted({row.source_record_id for row in rows if row.label == label and row.dataset == dataset})
        if not records: raise PairPlanError(f"no {label} records for {dataset}")
        ordered = sorted(records, key=lambda record: (_bucket(f"{label}|{record}"), record))
        cut = int(round(len(ordered) * TRAIN_FRACTION))
        cut = min(max(cut, 1), len(ordered) - 1) if len(ordered) > 1 else len(ordered)
        for position, record in enumerate(ordered):
            assignment[record] = "train" if position < cut else "validation"
    return assignment


def _subject_key(row: SourceRow) -> tuple[str, str] | None:
    return (row.dataset, row.subject_id) if row.subject_id else None


def _pick(candidates: list[SourceRow], live: SourceRow, count: int, salt: str) -> list[SourceRow]:
    """Deterministic, spread-out selection with the pairing rules enforced."""
    live_subject = _subject_key(live)
    usable = [row for row in candidates
              if row.source_record_id != live.source_record_id
              and (live_subject is None or _subject_key(row) is None or _subject_key(row) != live_subject)]
    if len(usable) < count:
        raise PairPlanError(f"only {len(usable)} usable spoof sources for live {live.sample_id} ({salt})")
    ordered = sorted(usable, key=lambda row: (_bucket(f"{salt}|{live.sample_id}|{row.sample_id}"), row.sample_id))
    chosen: list[SourceRow] = []
    seen_records: set[str] = set()
    for row in ordered:
        if row.source_record_id in seen_records: continue
        chosen.append(row); seen_records.add(row.source_record_id)
        if len(chosen) == count: return chosen
    for row in ordered:                       # relax the distinct-record preference if the pool is small
        if row not in chosen:
            chosen.append(row)
            if len(chosen) == count: return chosen
    raise PairPlanError(f"could not select {count} spoof sources for live {live.sample_id} ({salt})")


def build_pair_plan(package_root: Path, bank_root: Path, *, seed: int = PAIR_PLAN_SEED) -> dict[str, Any]:
    """Materialize the deterministic source-only GPAT pair plan."""
    package_lock = json.loads((Path(package_root) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    package_identity = str(package_lock["content_identity_sha256"])
    bank = load_bank(bank_root)
    bank_identity = str(bank["lock"]["bank_content_identity_sha256"])
    recipes = bank["recipes"]
    rows = load_source_train_rows(package_root)
    live_rows = [row for row in rows if row.label == "live"]
    spoof_rows = [row for row in rows if row.label == "spoof"]
    if not live_rows or not spoof_rows: raise PairPlanError("source_train must contain both live and spoof rows")
    live_partition = partition_records(rows, "live")
    spoof_partition = partition_records(rows, "spoof")
    subjects_available = all(row.subject_id for row in rows)

    pairs: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for live in live_rows:
        partition = live_partition[live.source_record_id]
        pool = [row for row in spoof_rows if spoof_partition[row.source_record_id] == partition]
        same = _pick([row for row in pool if row.dataset == live.dataset], live, SAME_DOMAIN_PER_LIVE, f"same|{partition}")
        cross = _pick([row for row in pool if row.dataset != live.dataset], live, CROSS_DOMAIN_PER_LIVE, f"cross|{partition}")
        for slot, (spoof, relation) in enumerate([(row, "same_domain") for row in same]
                                                 + [(row, "cross_domain") for row in cross]):
            recipe = recipes[_bucket(f"recipe|{partition}|{live.sample_id}|{spoof.sample_id}|{slot}|{seed}", len(recipes))]
            pair_id = "gpatpair_" + _digest(package_identity, bank_identity, partition, live.sample_id,
                                            spoof.sample_id, recipe.recipe_id, seed)[:20]
            pairs[partition].append({
                "pair_id": pair_id, "partition": partition, "slot": slot, "domain_relation": relation,
                "live_sample_id": live.sample_id, "live_dataset": live.dataset,
                "live_source_record_id": live.source_record_id,
                "spoof_sample_id": spoof.sample_id, "spoof_dataset": spoof.dataset,
                "spoof_source_record_id": spoof.source_record_id,
                "recipe_id": recipe.recipe_id, "recipe_seed": int(recipe.seed),
                "different_subject_rule": "enforced" if (_subject_key(live) and _subject_key(spoof)) else "not_applicable",
                "package_identity": package_identity, "recipe_bank_identity": bank_identity})
    for partition in pairs: pairs[partition].sort(key=lambda row: row["pair_id"])
    _assert_isolation(pairs, live_partition, spoof_partition)
    return {"pairs": pairs, "package_identity": package_identity, "recipe_bank_identity": bank_identity,
            "live_partition": live_partition, "spoof_partition": spoof_partition,
            "subjects_available": subjects_available, "seed": int(seed), "rows": rows}


def _assert_isolation(pairs: dict[str, list[dict[str, Any]]], live_partition: dict[str, str],
                      spoof_partition: dict[str, str]) -> None:
    for role, assignment in (("live", live_partition), ("spoof", spoof_partition)):
        train = {record for record, part in assignment.items() if part == "train"}
        validation = {record for record, part in assignment.items() if part == "validation"}
        overlap = train & validation
        if overlap: raise PairPlanError(f"{role} records appear in both partitions: {sorted(overlap)[:5]}")
    for partition, rows in pairs.items():
        for row in rows:
            if row["live_source_record_id"] == row["spoof_source_record_id"]:
                raise PairPlanError(f"pair {row['pair_id']} reuses one record for both roles")
    ids = [row["pair_id"] for rows in pairs.values() for row in rows]
    if len(set(ids)) != len(ids): raise PairPlanError("duplicate pair ids")


_PAIR_FIELDS = [("pair_id", pa.string()), ("partition", pa.string()), ("slot", pa.int32()),
                ("domain_relation", pa.string()), ("live_sample_id", pa.string()), ("live_dataset", pa.string()),
                ("live_source_record_id", pa.string()), ("spoof_sample_id", pa.string()),
                ("spoof_dataset", pa.string()), ("spoof_source_record_id", pa.string()),
                ("recipe_id", pa.string()), ("recipe_seed", pa.int64()), ("different_subject_rule", pa.string()),
                ("package_identity", pa.string()), ("recipe_bank_identity", pa.string())]


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    schema = pa.schema(_PAIR_FIELDS)
    table = pa.Table.from_pydict({name: [row[name] for row in rows] for name, _ in _PAIR_FIELDS}, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="none", version="2.6", write_statistics=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows: out[str(row[key])] = out.get(str(row[key]), 0) + 1
        return dict(sorted(out.items()))
    return {"pairs": len(rows), "live_datasets": counts("live_dataset"), "spoof_datasets": counts("spoof_dataset"),
            "domain_relation": counts("domain_relation"), "different_subject_rule": counts("different_subject_rule"),
            "distinct_live_samples": len({row["live_sample_id"] for row in rows}),
            "distinct_spoof_samples": len({row["spoof_sample_id"] for row in rows}),
            "distinct_live_records": len({row["live_source_record_id"] for row in rows}),
            "distinct_spoof_records": len({row["spoof_source_record_id"] for row in rows}),
            "distinct_recipes": len({row["recipe_id"] for row in rows}),
            "recipe_coverage": counts("recipe_id")}


def write_pair_plan(package_root: Path, bank_root: Path, output_root: Path, *, seed: int = PAIR_PLAN_SEED,
                    config_hash: str = "", dry_run: bool = False) -> dict[str, Any]:
    plan = build_pair_plan(package_root, bank_root, seed=seed)
    train, validation = plan["pairs"]["train"], plan["pairs"]["validation"]
    if len(train) != EXPECTED_TRAIN_PAIRS or len(validation) != EXPECTED_VALIDATION_PAIRS:
        raise PairPlanError(f"pair counts {len(train)}/{len(validation)} != "
                            f"{EXPECTED_TRAIN_PAIRS}/{EXPECTED_VALIDATION_PAIRS}")
    summary = {"pair_plan_schema_version": PAIR_PLAN_SCHEMA_VERSION, "seed": int(seed),
               "package_identity": plan["package_identity"], "recipe_bank_identity": plan["recipe_bank_identity"],
               "train": summarize_pairs(train), "validation": summarize_pairs(validation),
               "total_pairs": len(train) + len(validation),
               "subjects_available": plan["subjects_available"],
               "different_subject_rule": "enforced" if plan["subjects_available"] else "not_applicable",
               # The M3B package carries no attack-family/device column for source data.
               "attack_family_balance": "unavailable",
               "attack_family_balance_note": "no packaged attack-family metadata; balanced by live target domain, "
                                             "spoof source domain, source_record_id and recipe attributes",
               "source_dev_opened": False, "target_test_opened": False,
               "manifests_opened": [f"manifests/{SOURCE_SPLIT}.parquet"]}
    if dry_run:
        return {"status": "dry_run", "written": [], "summary": summary}
    output_root = Path(output_root)
    train_hash = _write_parquet(output_root / "pair_manifest_train.parquet", train)
    validation_hash = _write_parquet(output_root / "pair_manifest_validation.parquet", validation)
    atomic_json_write(output_root / "pair_plan_summary.json", summary)
    record_sets = {
        "live_train": _digest(*sorted(record for record, part in plan["live_partition"].items() if part == "train")),
        "live_validation": _digest(*sorted(record for record, part in plan["live_partition"].items() if part == "validation")),
        "spoof_train": _digest(*sorted(record for record, part in plan["spoof_partition"].items() if part == "train")),
        "spoof_validation": _digest(*sorted(record for record, part in plan["spoof_partition"].items() if part == "validation"))}
    lock = {"pair_plan_schema_version": PAIR_PLAN_SCHEMA_VERSION, "seed": int(seed),
            "package_identity": plan["package_identity"], "recipe_bank_identity": plan["recipe_bank_identity"],
            "config_hash": config_hash, "train_pairs": len(train), "validation_pairs": len(validation),
            "record_set_hashes": record_sets,
            "pair_manifest_sha256": {"train": train_hash, "validation": validation_hash},
            "pair_id_set_sha256": {"train": _digest(*[row["pair_id"] for row in train]),
                                   "validation": _digest(*[row["pair_id"] for row in validation])},
            "domain_composition": {"train": summary["train"]["live_datasets"],
                                   "validation": summary["validation"]["live_datasets"],
                                   "train_relation": summary["train"]["domain_relation"],
                                   "validation_relation": summary["validation"]["domain_relation"]},
            "recipe_coverage": {"train": summary["train"]["distinct_recipes"],
                                "validation": summary["validation"]["distinct_recipes"]},
            "attack_family_balance": "unavailable"}
    lock["pair_plan_identity_sha256"] = _digest(json.dumps(lock, sort_keys=True, separators=(",", ":")))
    atomic_json_write(output_root / "PAIR_PLAN_LOCK.json", lock)
    return {"status": "created", "written": ["pair_manifest_train.parquet", "pair_manifest_validation.parquet",
                                             "pair_plan_summary.json", "PAIR_PLAN_LOCK.json"],
            "summary": summary, "lock": lock}


def load_pair_manifest(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(Path(path)).to_pydict()
    return [{name: table[name][index] for name, _ in _PAIR_FIELDS} for index in range(len(table["pair_id"]))]


def pair_plan_identity(output_root: Path) -> str:
    lock = json.loads((Path(output_root) / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))
    return str(lock["pair_plan_identity_sha256"])
