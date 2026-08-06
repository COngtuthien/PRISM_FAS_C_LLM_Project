from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from prism_fas.recipes.bank import load_bank
from prism_fas.utils.core import atomic_json_write
from .pair_plan import ALLOWED_DATASETS, SOURCE_SPLIT, load_source_train_rows
from .physics import PHYSICS_ENGINE_VERSION

CANDIDATE_PLAN_SCHEMA_VERSION = "m8-candidate-plan-v1"
CANDIDATE_SEED = 20260806
ROUTES = ("physics", "gpat")
PHYSICS_NONE = "physics-none"
EXPECTED_PER_ROUTE = 560
EXPECTED_TOTAL = 1120
# `pair_manifest_sha256`-style byte hashes and provenance stay out of the
# identity for the same reason as the pair plan: they are not portable.
IDENTITY_EXCLUDED_FIELDS = ("config_hash", "candidate_plan_identity_sha256", "identity_excluded_fields",
                            "candidate_manifest_sha256")


class CandidatePlanError(ValueError):
    """The deterministic candidate plan cannot be built as declared."""


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _bucket(value: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], "big") % modulus


def load_bank_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise CandidatePlanError("synthetic bank config must be a YAML mapping")
    per_live = payload.get("candidate_recipes_per_live", {})
    if set(per_live) != set(ROUTES): raise CandidatePlanError(f"candidate_recipes_per_live must define {list(ROUTES)}")
    if payload.get("source", {}).get("split") != SOURCE_SPLIT:
        raise CandidatePlanError(f"candidate source split must be {SOURCE_SPLIT!r}")
    return payload


def candidate_id(*, package_identity: str, bank_identity: str, route: str, live_sample_id: str,
                 spoof_sample_id: str | None, recipe_id: str, seed: int, generator_binding: str) -> str:
    """`syn_<24 hex>` bound to every input that can change the pixels.

    Carries no path, filename, subject id, timestamp or target token.
    """
    return "syn_" + _digest(package_identity, bank_identity, route, live_sample_id,
                            spoof_sample_id or PHYSICS_NONE, recipe_id, int(seed), generator_binding)[:24]


def build_candidate_plan(package_root: Path, bank_root: Path, config: dict[str, Any], *,
                         gpat_checkpoint_sha256: str, seed: int = CANDIDATE_SEED) -> dict[str, Any]:
    """Deterministic 1120-row plan over every source_train live sample."""
    package_identity = json.loads((Path(package_root) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))["content_identity_sha256"]
    bank = load_bank(bank_root)
    bank_identity = str(bank["lock"]["bank_content_identity_sha256"])
    recipes = bank["recipes"]
    rows = load_source_train_rows(package_root)
    live = [row for row in rows if row.label == "live"]
    spoof = [row for row in rows if row.label == "spoof"]
    if not live or not spoof: raise CandidatePlanError("source_train must contain live and spoof samples")
    per_live = {route: int(config["candidate_recipes_per_live"][route]) for route in ROUTES}
    plan: list[dict[str, Any]] = []
    for target in live:
        gpat_sources = _gpat_sources(target, spoof, seed=seed, count=per_live["gpat"])
        for route in ROUTES:
            for slot in range(per_live[route]):
                recipe = _recipe_for(recipes, route, target.sample_id, slot, seed)
                source = gpat_sources[slot] if route == "gpat" else None
                binding = gpat_checkpoint_sha256 if route == "gpat" else PHYSICS_ENGINE_VERSION
                plan.append({
                    "synthetic_id": candidate_id(package_identity=package_identity, bank_identity=bank_identity,
                                                 route=route, live_sample_id=target.sample_id,
                                                 spoof_sample_id=source.sample_id if source else None,
                                                 recipe_id=recipe.recipe_id, seed=seed, generator_binding=binding),
                    "route": route, "slot": slot,
                    "live_target_sample_id": target.sample_id, "live_target_dataset": target.dataset,
                    "live_target_record_id": target.source_record_id,
                    "spoof_source_sample_id": source.sample_id if source else None,
                    "spoof_source_dataset": source.dataset if source else None,
                    "spoof_source_record_id": source.source_record_id if source else None,
                    "domain_relation": ("same_domain" if source and source.dataset == target.dataset
                                        else "cross_domain" if source else "not_applicable"),
                    "recipe_id": recipe.recipe_id, "recipe_seed": int(recipe.seed),
                    "candidate_seed": int(seed),
                    "generator_binding": binding,
                    "gpat_checkpoint_sha256": gpat_checkpoint_sha256 if route == "gpat" else None,
                    "physics_engine_version": PHYSICS_ENGINE_VERSION if route == "physics" else None,
                    "package_identity": package_identity, "recipe_bank_identity": bank_identity})
    plan.sort(key=lambda row: row["synthetic_id"])
    _validate(plan, per_live, len(live))
    return {"rows": plan, "package_identity": package_identity, "recipe_bank_identity": bank_identity,
            "live_samples": len(live), "seed": int(seed),
            "gpat_checkpoint_sha256": gpat_checkpoint_sha256}


def _recipe_for(recipes: list[Any], route: str, live_sample_id: str, slot: int, seed: int) -> Any:
    """Coverage-aware deterministic assignment, WITHOUT replacement per
    (live sample, route).

    Drawing each slot independently collides for ~1/len(recipes) of samples, and
    a physics candidate carries no spoof source, so an identical recipe would
    make two candidate ids identical. Ordering the bank once per (route, live)
    and indexing by slot guarantees distinct recipes.
    """
    ordered = sorted(recipes, key=lambda recipe: (
        _bucket(f"candidate|{route}|{live_sample_id}|{seed}|{recipe.recipe_id}", 10 ** 9), recipe.recipe_id))
    return ordered[slot % len(ordered)]


def _gpat_sources(target: Any, spoof: list[Any], *, seed: int, count: int) -> list[Any]:
    """One same-domain and one cross-domain spoof source, different record and
    different explicit subject, deterministic."""
    chosen: list[Any] = []
    for index in range(count):
        want_same = index % 2 == 0
        pool = [row for row in spoof
                if (row.dataset == target.dataset) == want_same
                and row.source_record_id != target.source_record_id
                and not (row.subject_id and target.subject_id
                         and row.dataset == target.dataset and row.subject_id == target.subject_id)]
        if not pool: raise CandidatePlanError(f"no {'same' if want_same else 'cross'}-domain spoof source for {target.sample_id}")
        ordered = sorted(pool, key=lambda row: (_bucket(f"cand|{seed}|{target.sample_id}|{index}|{row.sample_id}", 10 ** 9),
                                                row.sample_id))
        pick = next((row for row in ordered if row.sample_id not in {item.sample_id for item in chosen}), ordered[0])
        chosen.append(pick)
    return chosen


def _validate(plan: list[dict[str, Any]], per_live: dict[str, int], live_count: int) -> None:
    if len(plan) != live_count * sum(per_live.values()):
        raise CandidatePlanError(f"plan holds {len(plan)} rows, expected {live_count * sum(per_live.values())}")
    ids = [row["synthetic_id"] for row in plan]
    if len(set(ids)) != len(ids): raise CandidatePlanError("duplicate synthetic ids")
    for route in ROUTES:
        count = sum(1 for row in plan if row["route"] == route)
        if count != live_count * per_live[route]:
            raise CandidatePlanError(f"route {route} has {count} rows, expected {live_count * per_live[route]}")
    for row in plan:
        if row["route"] == "physics" and row["spoof_source_sample_id"] is not None:
            raise CandidatePlanError("a physics candidate must not carry a spoof source")
        if row["route"] == "gpat":
            if row["spoof_source_sample_id"] is None: raise CandidatePlanError("a gpat candidate needs a spoof source")
            if row["spoof_source_record_id"] == row["live_target_record_id"]:
                raise CandidatePlanError("a gpat candidate reuses one record for both roles")
        if row["live_target_dataset"] not in ALLOWED_DATASETS:
            raise CandidatePlanError(f"unexpected dataset {row['live_target_dataset']!r}")
    relations = {row["domain_relation"] for row in plan if row["route"] == "gpat"}
    if relations != {"same_domain", "cross_domain"}:
        raise CandidatePlanError(f"gpat plan must contain both domain relations, got {sorted(relations)}")


_FIELDS = [("synthetic_id", pa.string()), ("route", pa.string()), ("slot", pa.int32()),
           ("live_target_sample_id", pa.string()), ("live_target_dataset", pa.string()),
           ("live_target_record_id", pa.string()), ("spoof_source_sample_id", pa.string()),
           ("spoof_source_dataset", pa.string()), ("spoof_source_record_id", pa.string()),
           ("domain_relation", pa.string()), ("recipe_id", pa.string()), ("recipe_seed", pa.int64()),
           ("candidate_seed", pa.int64()), ("generator_binding", pa.string()),
           ("gpat_checkpoint_sha256", pa.string()), ("physics_engine_version", pa.string()),
           ("package_identity", pa.string()), ("recipe_bank_identity", pa.string())]


def rows_digest(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps([{name: row[name] for name, _ in _FIELDS} for row in rows],
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_candidate_plan(package_root: Path, bank_root: Path, config_path: Path, output_root: Path, *,
                         gpat_checkpoint_sha256: str, seed: int = CANDIDATE_SEED,
                         dry_run: bool = False) -> dict[str, Any]:
    config = load_bank_config(config_path)
    plan = build_candidate_plan(package_root, bank_root, config, gpat_checkpoint_sha256=gpat_checkpoint_sha256, seed=seed)
    rows = plan["rows"]
    if len(rows) != EXPECTED_TOTAL: raise CandidatePlanError(f"expected {EXPECTED_TOTAL} candidates, got {len(rows)}")
    summary = _summary(rows, plan)
    if dry_run: return {"status": "dry_run", "written": [], "summary": summary}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict({name: [row[name] for row in rows] for name, _ in _FIELDS}, schema=pa.schema(_FIELDS))
    target = output_root / "candidate_plan.parquet"
    pq.write_table(table, target, compression="none", version="2.6", write_statistics=False)
    lock = {"candidate_plan_schema_version": CANDIDATE_PLAN_SCHEMA_VERSION, "seed": int(seed),
            "package_identity": plan["package_identity"], "recipe_bank_identity": plan["recipe_bank_identity"],
            "gpat_checkpoint_sha256": plan["gpat_checkpoint_sha256"],
            "physics_engine_version": PHYSICS_ENGINE_VERSION,
            "candidate_count": len(rows), "route_counts": summary["route_counts"],
            "live_samples": plan["live_samples"],
            "candidate_rows_sha256": rows_digest(rows),
            "candidate_id_set_sha256": _digest(*[row["synthetic_id"] for row in rows]),
            "candidate_manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "domain_relation_counts": summary["domain_relation_counts"],
            "live_target_datasets": summary["live_target_datasets"],
            "distinct_recipes": summary["distinct_recipes"],
            "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()}
    lock["candidate_plan_identity_sha256"] = _digest(json.dumps(
        {key: value for key, value in lock.items() if key not in IDENTITY_EXCLUDED_FIELDS},
        sort_keys=True, separators=(",", ":")))
    lock["identity_excluded_fields"] = list(IDENTITY_EXCLUDED_FIELDS)
    atomic_json_write(output_root / "CANDIDATE_PLAN_LOCK.json", lock)
    return {"status": "created", "written": ["candidate_plan.parquet", "CANDIDATE_PLAN_LOCK.json"],
            "summary": summary, "lock": lock, "rows": rows}


def _summary(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    def counts(key: str, subset: list[dict[str, Any]] | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in (subset if subset is not None else rows):
            value = row[key]
            if value is None: continue
            out[str(value)] = out.get(str(value), 0) + 1
        return dict(sorted(out.items()))
    gpat = [row for row in rows if row["route"] == "gpat"]
    return {"candidates": len(rows), "route_counts": counts("route"),
            "live_target_datasets": counts("live_target_dataset"),
            "spoof_source_datasets": counts("spoof_source_dataset", gpat),
            "domain_relation_counts": counts("domain_relation", gpat),
            "distinct_live_samples": len({row["live_target_sample_id"] for row in rows}),
            "distinct_recipes": len({row["recipe_id"] for row in rows}),
            "gpat_checkpoint_sha256": plan["gpat_checkpoint_sha256"],
            "physics_engine_version": PHYSICS_ENGINE_VERSION}


def load_candidate_plan(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(Path(path)).to_pydict()
    return [{name: table[name][index] for name, _ in _FIELDS} for index in range(len(table["synthetic_id"]))]
