from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any
import yaml
from .audit import bank_audit
from .canonical import recipe_hash, text_hash
from .compile import COMPILER_VERSION, compile_recipes, compile_summary
from .conditioning import CONDITIONING_DIM, CONDITIONING_VERSION, feature_names_sha256
from .generate import (GENERATOR_EXTERNAL_LLM_INVOKED, GENERATOR_MODEL_ID, GENERATOR_PROVIDER, GENERATOR_REVISION,
                       generate_recipes, generator_manifest, recipes_from_jsonl, recipes_to_jsonl, render_prompt)
from .ontology import Ontology, load_ontology, parse_ontology
from .schema import RecipeV11
from .validate import validate_recipes

BANK_SCHEMA_VERSION = "m7-bank-v1"
BANK_STATUS_FROZEN = "frozen"
BANK_FILES = ("recipes.jsonl", "ontology.yaml", "prompt.txt", "generator.json", "coverage.json",
              "validation.json", "BANK_LOCK.json")
CONTENT_FILES = ("recipes.jsonl", "ontology.yaml", "prompt.txt", "generator.json", "coverage.json", "validation.json")


class BankError(RuntimeError):
    """A frozen recipe bank is missing, inconsistent or would be overwritten."""


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def load_bank_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise BankError("bank config must be a YAML mapping")
    missing = [key for key in ("bank_schema_version", "bank_id", "recipe_count", "bank_seed") if key not in payload]
    if missing: raise BankError(f"bank config is missing keys: {missing}")
    if str(payload["bank_schema_version"]) != BANK_SCHEMA_VERSION:
        raise BankError(f"bank config schema {payload['bank_schema_version']!r} != {BANK_SCHEMA_VERSION!r}")
    return payload


def bank_content_identity(*, bank_id: str, recipe_count: int, bank_seed: int, file_hashes: dict[str, str],
                          recipe_hashes: dict[str, str], graph_hashes: dict[str, str],
                          ontology_version: str, compiler_version: str, conditioning: dict[str, Any]) -> str:
    """Identity over bank content only. No wall-clock, host, machine path or
    build duration participates, so a rebuild on another machine reproduces it."""
    payload = {"bank_schema_version": BANK_SCHEMA_VERSION, "bank_id": bank_id, "recipe_count": int(recipe_count),
               "bank_seed": int(bank_seed), "file_sha256": dict(sorted(file_hashes.items())),
               "recipe_hashes": dict(sorted(recipe_hashes.items())), "graph_hashes": dict(sorted(graph_hashes.items())),
               "ontology_version": ontology_version, "compiler_version": compiler_version,
               "generator": {"provider": GENERATOR_PROVIDER, "model_id": GENERATOR_MODEL_ID,
                             "revision": GENERATOR_REVISION, "external_llm_invoked": GENERATOR_EXTERNAL_LLM_INVOKED},
               "conditioning": dict(sorted(conditioning.items()))}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_payloads(ontology: Ontology, *, bank_id: str, recipe_count: int, bank_seed: int) -> dict[str, Any]:
    recipes = generate_recipes(ontology, count=recipe_count, bank_id=bank_id, bank_seed=bank_seed)
    validation = validate_recipes(recipes, ontology)
    if not validation["passed"]:
        raise BankError(f"generated bank failed validation: {json.dumps(validation['issues'][:5])}")
    graphs = compile_recipes(recipes, ontology, bank_id=bank_id)
    audit = bank_audit(recipes, ontology)
    if not audit["passed"]:
        raise BankError(f"generated bank failed the coverage/diversity audit: {json.dumps(audit['coverage']['required'])}")
    compiled = compile_summary(graphs)
    if compiled["duplicate_graph_hashes"]:
        raise BankError(f"compiled graphs collide: {compiled['duplicate_graph_hashes']}")
    texts = {
        "recipes.jsonl": recipes_to_jsonl(recipes),
        "ontology.yaml": ontology.source_text,
        "prompt.txt": render_prompt(ontology),
        "generator.json": _json_text(generator_manifest(bank_id=bank_id, bank_seed=bank_seed, count=recipe_count, ontology=ontology)),
        "coverage.json": _json_text({"bank_id": bank_id, **audit, "compiled": compiled}),
        "validation.json": _json_text({"bank_id": bank_id, **validation})}
    return {"recipes": recipes, "graphs": graphs, "validation": validation, "audit": audit,
            "compiled": compiled, "texts": texts}


def _lock_payload(ontology: Ontology, *, bank_id: str, bank_seed: int, recipes: list[RecipeV11],
                  graphs: list[Any], texts: dict[str, str]) -> dict[str, Any]:
    file_hashes = {name: text_hash(texts[name]) for name in CONTENT_FILES}
    recipe_hashes = {recipe.recipe_id: recipe_hash(recipe) for recipe in recipes}
    graph_hashes = {graph.recipe_id: graph.graph_hash for graph in graphs}
    conditioning = {"version": CONDITIONING_VERSION, "dimension": CONDITIONING_DIM,
                    "feature_names_sha256": feature_names_sha256(ontology)}
    identity = bank_content_identity(bank_id=bank_id, recipe_count=len(recipes), bank_seed=bank_seed,
                                     file_hashes=file_hashes, recipe_hashes=recipe_hashes, graph_hashes=graph_hashes,
                                     ontology_version=ontology.version, compiler_version=COMPILER_VERSION,
                                     conditioning=conditioning)
    return {"bank_schema_version": BANK_SCHEMA_VERSION, "bank_id": bank_id, "status": BANK_STATUS_FROZEN,
            "recipe_count": len(recipes), "bank_seed": int(bank_seed),
            "generator": {"provider": GENERATOR_PROVIDER, "model_id": GENERATOR_MODEL_ID,
                          "revision": GENERATOR_REVISION, "external_llm_invoked": GENERATOR_EXTERNAL_LLM_INVOKED},
            "ontology_version": ontology.version, "ontology_sha256": file_hashes["ontology.yaml"],
            "prompt_sha256": file_hashes["prompt.txt"], "recipes_jsonl_sha256": file_hashes["recipes.jsonl"],
            "coverage_sha256": file_hashes["coverage.json"], "validation_sha256": file_hashes["validation.json"],
            "generator_sha256": file_hashes["generator.json"], "file_sha256": file_hashes,
            "compiler_version": COMPILER_VERSION, "conditioning": conditioning,
            "recipe_hashes": recipe_hashes, "graph_hashes": graph_hashes,
            "bank_content_identity_sha256": identity}


def build_bank(output_root: Path, ontology_path: Path, config_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Build or reuse a frozen recipe bank.

    Rebuilding with identical inputs is a no-op: the existing files are proven
    byte-identical and left untouched. A destination holding a *different* lock
    is never overwritten — a new versioned bank directory is required instead.
    """
    config = load_bank_config(config_path)
    ontology = load_ontology(ontology_path)
    bank_id, recipe_count, bank_seed = str(config["bank_id"]), int(config["recipe_count"]), int(config["bank_seed"])
    built = _build_payloads(ontology, bank_id=bank_id, recipe_count=recipe_count, bank_seed=bank_seed)
    texts = built["texts"]
    lock = _lock_payload(ontology, bank_id=bank_id, bank_seed=bank_seed, recipes=built["recipes"],
                         graphs=built["graphs"], texts=texts)
    texts_with_lock = {**texts, "BANK_LOCK.json": _json_text(lock)}
    root = Path(output_root)
    plan = {"bank_id": bank_id, "recipe_count": recipe_count, "bank_seed": bank_seed, "output_root": root.as_posix(),
            "ontology_version": ontology.version, "ontology_sha256": ontology.sha256,
            "bank_content_identity_sha256": lock["bank_content_identity_sha256"],
            "files": list(BANK_FILES), "compiler_version": COMPILER_VERSION,
            "conditioning_version": CONDITIONING_VERSION, "conditioning_dimension": CONDITIONING_DIM,
            "external_llm_invoked": GENERATOR_EXTERNAL_LLM_INVOKED}
    existing_lock_path = root / "BANK_LOCK.json"
    if existing_lock_path.is_file():
        existing = json.loads(existing_lock_path.read_text(encoding="utf-8"))
        if existing.get("bank_content_identity_sha256") != lock["bank_content_identity_sha256"]:
            raise BankError(
                f"{root.as_posix()} already holds bank {existing.get('bank_id')!r} with identity "
                f"{existing.get('bank_content_identity_sha256')} != {lock['bank_content_identity_sha256']}; "
                "banks are immutable, create a new versioned bank directory instead")
        differing = [name for name in BANK_FILES
                     if not (root / name).is_file() or (root / name).read_text(encoding="utf-8") != texts_with_lock[name]]
        if differing:
            raise BankError(f"{root.as_posix()} has a matching lock but differing files {differing}; refusing to rewrite a frozen bank")
        return {"status": "reused", "written": [], **plan, "audit": built["audit"], "validation": built["validation"]}
    if dry_run:
        return {"status": "dry_run", "written": [], **plan, "audit": built["audit"], "validation": built["validation"]}
    for name in BANK_FILES:
        _atomic_text(root / name, texts_with_lock[name])
    return {"status": "created", "written": list(BANK_FILES), **plan, "audit": built["audit"], "validation": built["validation"]}


def load_bank(root: Path) -> dict[str, Any]:
    root = Path(root)
    missing = [name for name in BANK_FILES if not (root / name).is_file()]
    if missing: raise BankError(f"{root.as_posix()} is not a frozen recipe bank; missing {missing}")
    lock = json.loads((root / "BANK_LOCK.json").read_text(encoding="utf-8"))
    ontology = parse_ontology((root / "ontology.yaml").read_text(encoding="utf-8"), path=root / "ontology.yaml")
    recipes = recipes_from_jsonl((root / "recipes.jsonl").read_text(encoding="utf-8"))
    return {"root": root, "lock": lock, "ontology": ontology, "recipes": recipes,
            "bank_id": str(lock["bank_id"]), "prompt": (root / "prompt.txt").read_text(encoding="utf-8")}


def validate_bank(root: Path) -> dict[str, Any]:
    """Re-derive every hash in BANK_LOCK.json from the files on disk."""
    bank = load_bank(root)
    root, lock, ontology, recipes = Path(bank["root"]), bank["lock"], bank["ontology"], bank["recipes"]
    errors: list[str] = []
    if lock.get("bank_schema_version") != BANK_SCHEMA_VERSION: errors.append("bank_schema_version mismatch")
    if lock.get("status") != BANK_STATUS_FROZEN: errors.append(f"status {lock.get('status')!r} != {BANK_STATUS_FROZEN!r}")
    if int(lock.get("recipe_count", -1)) != len(recipes): errors.append("recipe_count does not match recipes.jsonl")
    file_hashes = {name: text_hash((root / name).read_text(encoding="utf-8")) for name in CONTENT_FILES}
    for name, digest in file_hashes.items():
        if lock.get("file_sha256", {}).get(name) != digest: errors.append(f"{name} sha256 mismatch")
    if lock.get("ontology_sha256") != file_hashes["ontology.yaml"]: errors.append("ontology_sha256 mismatch")
    if lock.get("prompt_sha256") != file_hashes["prompt.txt"]: errors.append("prompt_sha256 mismatch")
    if lock.get("recipes_jsonl_sha256") != file_hashes["recipes.jsonl"]: errors.append("recipes_jsonl_sha256 mismatch")
    if lock.get("coverage_sha256") != file_hashes["coverage.json"]: errors.append("coverage_sha256 mismatch")
    if lock.get("validation_sha256") != file_hashes["validation.json"]: errors.append("validation_sha256 mismatch")
    if lock.get("generator", {}).get("external_llm_invoked") is not False: errors.append("generator claims an external LLM was invoked")
    validation = validate_recipes(recipes, ontology)
    if not validation["passed"]: errors.append(f"{validation['issue_count']} recipe validation issues")
    graphs = compile_recipes(recipes, ontology, bank_id=str(lock["bank_id"]))
    recipe_hashes = {recipe.recipe_id: recipe_hash(recipe) for recipe in recipes}
    graph_hashes = {graph.recipe_id: graph.graph_hash for graph in graphs}
    if lock.get("recipe_hashes") != recipe_hashes: errors.append("recipe_hashes mismatch")
    if lock.get("graph_hashes") != graph_hashes: errors.append("graph_hashes mismatch")
    conditioning = {"version": CONDITIONING_VERSION, "dimension": CONDITIONING_DIM,
                    "feature_names_sha256": feature_names_sha256(ontology)}
    if lock.get("conditioning") != conditioning: errors.append("conditioning contract mismatch")
    if lock.get("compiler_version") != COMPILER_VERSION: errors.append("compiler_version mismatch")
    identity = bank_content_identity(bank_id=str(lock["bank_id"]), recipe_count=len(recipes),
                                     bank_seed=int(lock["bank_seed"]), file_hashes=file_hashes,
                                     recipe_hashes=recipe_hashes, graph_hashes=graph_hashes,
                                     ontology_version=ontology.version, compiler_version=COMPILER_VERSION,
                                     conditioning=conditioning)
    if lock.get("bank_content_identity_sha256") != identity: errors.append("bank_content_identity_sha256 mismatch")
    audit = bank_audit(recipes, ontology)
    if not audit["passed"]: errors.append("coverage/diversity audit failed")
    # Only the directory name is reported: a validation report is a committed /
    # audited artifact and must never carry a machine-specific absolute path.
    return {"bank_root_name": root.name, "bank_id": str(lock["bank_id"]), "passed": not errors, "errors": errors,
            "recipe_count": len(recipes), "bank_content_identity_sha256": identity,
            "ontology_version": ontology.version, "ontology_sha256": file_hashes["ontology.yaml"],
            "compiler_version": COMPILER_VERSION, "conditioning": conditioning,
            "validation": validation, "coverage": audit["coverage"], "diversity": audit["diversity"],
            "compiled": compile_summary(graphs), "external_llm_invoked": bool(lock.get("generator", {}).get("external_llm_invoked"))}


def compiled_graphs(root: Path) -> dict[str, Any]:
    bank = load_bank(root)
    graphs = compile_recipes(bank["recipes"], bank["ontology"], bank_id=bank["bank_id"])
    return {"bank_id": bank["bank_id"], "graphs": graphs, "summary": compile_summary(graphs),
            "ontology": bank["ontology"], "recipes": bank["recipes"], "lock": bank["lock"]}
