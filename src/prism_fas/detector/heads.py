"""M9 GlobalHead, PromptHead and the cached frozen recipe text embeddings.

Spec section 9.2 (Table 32) gives the two heads:

    p_global = GlobalHead(z_global)
    p_prompt = PromptHead(z_r, frozen_recipe_text_embeddings)

and spec section 10.1 (Table 35) gives PromptHead's only supervision:

    L_prompt = InfoNCE(z_attack_region, text_embedding(recipe))

The recipe text embeddings are produced ONCE, offline, by the pinned frozen SigLIP2
text encoder over the frozen M7 bank's own deterministic recipe description
(`prism_fas.recipes.canonical.recipe_description`). At training and inference time
they are a constant buffer: no network call, no online text generation, no LLM, and
no target taxonomy text ever enters this path.

The M8 quality-gate row `recipe_match = not_applicable` is an M8 gate placeholder
recorded because no PromptHead existed then. It is NOT a training target and is
never read here; a test asserts that.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from .contracts import REGION_COUNT, DetectorContractError
from .npz_io import read_arrays_npz, write_arrays_npz
from .pretrained import SIGLIP2_PIN, SIGLIP2_TOKENIZATION, SigLIP2Artifacts

HEADS_SCHEMA_VERSION = "m9-heads-v1"
TEXT_CACHE_SCHEMA_VERSION = "m9-recipe-text-cache-v1"
# The exact recipe text source. Bound into the cache identity so swapping the
# description function invalidates every cache built from the old one.
RECIPE_TEXT_SOURCE = "prism_fas.recipes.canonical.recipe_description"
RECIPE_TEXT_SOURCE_VERSION = "m7-recipe-description-v1"
# SPEC_UNDERSPECIFIED, docs/M9_DETECTOR_CONTRACT.md section 4.
DEFAULT_PROMPT_TEMPERATURE = 0.07
EMBEDDING_ROUND_DECIMALS = 6


class TextCacheError(RuntimeError):
    """The recipe text cache is missing, malformed or bound to different inputs."""


# --- GlobalHead -------------------------------------------------------------

class GlobalHead(nn.Module):
    """`p_global = GlobalHead(z_global)`.

    SPEC_UNDERSPECIFIED shape: the spec names the head but not its width, so it is
    the smallest thing that type-checks — one linear layer to a single logit, using
    the M5 convention (live=0, spoof=1) so `CE` is BCE-with-logits.
    """

    def __init__(self, global_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.global_dim = int(global_dim)
        self.norm = nn.LayerNorm(self.global_dim)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(self.global_dim, 1)
        nn.init.zeros_(self.linear.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.01)

    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """`[B,Dg]` -> `[B,1]` logit."""
        if global_embedding.dim() != 2 or global_embedding.shape[1] != self.global_dim:
            raise DetectorContractError(f"GlobalHead expects [B,{self.global_dim}], got {tuple(global_embedding.shape)}")
        return self.linear(self.dropout(self.norm(global_embedding)))


# --- cached frozen recipe text embeddings -----------------------------------

@dataclass(frozen=True)
class RecipeTextCache:
    """Frozen recipe text embeddings, one row per recipe in the M7 bank.

    `identity` binds the recipe bank, the exact text source, the SigLIP2 pin, the
    tokenization contract, the embedding dimension, the normalization and the
    embedding content. It binds no absolute path, no machine name and no timestamp.
    """
    recipe_ids: tuple[str, ...]
    embeddings: np.ndarray                 # [N, text_dim] float32, L2-normalized
    identity: str
    binding: dict[str, Any]

    def validate(self) -> "RecipeTextCache":
        if self.embeddings.ndim != 2: raise TextCacheError("recipe text embeddings must be [N,D]")
        if self.embeddings.shape[0] != len(self.recipe_ids):
            raise TextCacheError("recipe text embeddings and recipe ids disagree on N")
        if not np.isfinite(self.embeddings).all(): raise TextCacheError("recipe text embeddings are not finite")
        norms = np.linalg.norm(self.embeddings.astype(np.float64), axis=1)
        if float(np.abs(norms - 1.0).max()) > 1e-3:
            raise TextCacheError("recipe text embeddings are not L2-normalized")
        if len(set(self.recipe_ids)) != len(self.recipe_ids):
            raise TextCacheError("duplicate recipe ids in the text cache")
        return self

    @property
    def count(self) -> int: return int(self.embeddings.shape[0])
    @property
    def dim(self) -> int: return int(self.embeddings.shape[1])

    def tensor(self, *, device: str = "cpu", dtype: Any = torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.embeddings, dtype=dtype, device=device)

    def index_of(self, recipe_id: str) -> int:
        try: return self.recipe_ids.index(recipe_id)
        except ValueError: raise KeyError(f"recipe {recipe_id!r} is not in the frozen text cache") from None


def content_identity(embeddings: np.ndarray, recipe_ids: tuple[str, ...]) -> str:
    """Content digest of the cached matrix. Rounded so a benign float-repr
    difference cannot pretend to be a different cache."""
    payload = {"recipe_ids": list(recipe_ids),
               "embeddings": np.round(np.asarray(embeddings, dtype=np.float64),
                                      EMBEDDING_ROUND_DECIMALS).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def cache_identity(binding: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def recipe_texts(bank: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(recipe_ids, descriptions)` in the bank's own recipe order.

    Uses the frozen bank's deterministic description; no text is generated here and
    no model is invoked to write it.
    """
    from prism_fas.recipes.canonical import recipe_description
    recipes = list(bank["recipes"])
    return (tuple(recipe.recipe_id for recipe in recipes),
            tuple(recipe_description(recipe) for recipe in recipes))


def _token_ids(tokenizer: Any, texts: tuple[str, ...]) -> np.ndarray:
    encoded = tokenizer(list(texts), padding=SIGLIP2_TOKENIZATION["padding"],
                        max_length=int(SIGLIP2_TOKENIZATION["max_length"]),
                        truncation=bool(SIGLIP2_TOKENIZATION["truncation"]),
                        add_special_tokens=bool(SIGLIP2_TOKENIZATION["add_special_tokens"]))
    return np.asarray(encoded["input_ids"], dtype=np.int64)


def build_recipe_text_cache(bank_root: Path, artifacts: SigLIP2Artifacts, *,
                            device: str = "cpu", batch_size: int = 32) -> RecipeTextCache:
    """Encode every frozen recipe description once with the frozen SigLIP2 text
    tower and L2-normalize the result.

    Deterministic: fixed recipe order, fixed tokenization, eval mode, no dropout,
    no sampling, fp32.
    """
    from prism_fas.recipes.bank import load_bank
    bank = load_bank(Path(bank_root))
    bank_identity = str(bank["lock"]["bank_content_identity_sha256"])
    ids, texts = recipe_texts(bank)
    tokenizer = artifacts.load_tokenizer()
    if type(tokenizer).__name__ != SIGLIP2_PIN["tokenizer_class"]:
        raise TextCacheError(f"tokenizer is {type(tokenizer).__name__}, pinned {SIGLIP2_PIN['tokenizer_class']}")
    token_ids = _token_ids(tokenizer, texts)
    model = artifacts.load_model(device=device)
    text_model = getattr(model, "text_model", None)
    if text_model is None: raise TextCacheError("the pinned SigLIP2 model exposes no text tower")
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, token_ids.shape[0], int(batch_size)):
            chunk = torch.as_tensor(token_ids[start:start + int(batch_size)], device=device)
            pooled = text_model(input_ids=chunk).pooler_output
            rows.append(torch.nn.functional.normalize(pooled.float(), dim=-1, eps=1e-6).cpu().numpy())
    embeddings = np.ascontiguousarray(np.concatenate(rows, axis=0).astype(np.float32))
    if embeddings.shape[1] != int(SIGLIP2_PIN["text_hidden_size"]):
        raise TextCacheError(f"text embedding dim {embeddings.shape[1]} != pinned {SIGLIP2_PIN['text_hidden_size']}")
    binding = _binding(bank_identity=bank_identity, bank_id=str(bank["bank_id"]), recipe_ids=ids,
                       texts=texts, token_ids=token_ids, artifacts=artifacts, embeddings=embeddings)
    return RecipeTextCache(recipe_ids=ids, embeddings=embeddings,
                           identity=cache_identity(binding), binding=binding).validate()


def _binding(*, bank_identity: str, bank_id: str, recipe_ids: tuple[str, ...], texts: tuple[str, ...],
             token_ids: np.ndarray, artifacts: SigLIP2Artifacts, embeddings: np.ndarray) -> dict[str, Any]:
    """Everything the cache identity must bind — and nothing it must not.

    Deliberately absent: absolute path, machine name, timestamp, user, device.
    """
    identity_files = SIGLIP2_PIN["text_identity_files"]
    return {
        "schema_version": TEXT_CACHE_SCHEMA_VERSION,
        "recipe_bank_id": bank_id,
        "recipe_bank_identity_sha256": bank_identity,
        "recipe_count": len(recipe_ids),
        "recipe_ids_sha256": hashlib.sha256("|".join(recipe_ids).encode("utf-8")).hexdigest(),
        "text_source": RECIPE_TEXT_SOURCE,
        "text_source_version": RECIPE_TEXT_SOURCE_VERSION,
        "text_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
        "model_id": SIGLIP2_PIN["model_id"],
        "model_revision": SIGLIP2_PIN["revision"],
        "model_architecture": SIGLIP2_PIN["architecture"],
        "model_file_sha256": {name: artifacts.digests[name] for name in sorted(identity_files)},
        "tokenizer_class": SIGLIP2_PIN["tokenizer_class"],
        "tokenization": dict(sorted(SIGLIP2_TOKENIZATION.items())),
        "token_ids_sha256": hashlib.sha256(np.ascontiguousarray(token_ids).tobytes()).hexdigest(),
        "token_ids_shape": list(token_ids.shape),
        "embedding_dim": int(embeddings.shape[1]),
        "pooling": "text_model.pooler_output",
        "normalization": "l2",
        "dtype": "float32",
        "content_identity_sha256": content_identity(embeddings, recipe_ids)}


def write_recipe_text_cache(path: Path, cache: RecipeTextCache) -> dict[str, Any]:
    """Arrays only, so the cache loads with `allow_pickle=False`."""
    cache.validate()
    arrays = {"recipe_ids": np.array(list(cache.recipe_ids), dtype="U32"),
              "embeddings": cache.embeddings.astype(np.float32),
              "cache_identity": np.array([cache.identity], dtype="U64"),
              "binding_json": np.frombuffer(
                  json.dumps(cache.binding, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                  dtype=np.uint8),
              "schema_version": np.array([TEXT_CACHE_SCHEMA_VERSION], dtype="U40")}
    written = write_arrays_npz(Path(path), arrays)
    return {**written, "cache_identity_sha256": cache.identity, "recipe_count": cache.count,
            "embedding_dim": cache.dim}


RECIPE_TEXT_CACHE_FILENAME = "recipe_text_cache.npz"
# Candidate locations, relative to a weight root, in the order they are tried.
RECIPE_TEXT_CACHE_RELPATHS = (RECIPE_TEXT_CACHE_FILENAME,
                              f"m9/{RECIPE_TEXT_CACHE_FILENAME}",
                              f"pretrained/m9/{RECIPE_TEXT_CACHE_FILENAME}")


def resolve_recipe_text_cache(weight_root: Path, *, expected_identity: str | None = None) -> RecipeTextCache:
    """Load the FROZEN recipe text cache artifact that ships beside the SigLIP2 pin.

    The cache is an artifact, not a recomputation. Encoding 128 descriptions with a
    frozen tower is deterministic within one environment but not bit-identical
    across torch/transformers builds or across CPU and GPU, so recomputing it
    remotely would produce a different content identity for the same science. It is
    therefore built once, uploaded next to the weights, and only ever verified.
    """
    root = Path(weight_root)
    path = next((root / relative for relative in RECIPE_TEXT_CACHE_RELPATHS
                 if (root / relative).is_file()), None)
    if path is None:
        raise TextCacheError(f"the frozen recipe text cache is missing under {root.name}; "
                             f"looked for {list(RECIPE_TEXT_CACHE_RELPATHS)}")
    return read_recipe_text_cache(path, expected_identity=expected_identity)


def read_recipe_text_cache(path: Path, *, expected_identity: str | None = None) -> RecipeTextCache:
    """Load and re-derive. The stored identity is recomputed from the stored
    binding, so a hand-edited cache cannot claim an identity it does not have."""
    arrays = read_arrays_npz(Path(path))
    if str(arrays["schema_version"][0]) != TEXT_CACHE_SCHEMA_VERSION:
        raise TextCacheError(f"text cache schema {arrays['schema_version'][0]!r} != {TEXT_CACHE_SCHEMA_VERSION!r}")
    binding = json.loads(bytes(arrays["binding_json"].astype(np.uint8)).decode("utf-8"))
    cache = RecipeTextCache(recipe_ids=tuple(str(value) for value in arrays["recipe_ids"]),
                            embeddings=np.asarray(arrays["embeddings"], dtype=np.float32),
                            identity=str(arrays["cache_identity"][0]), binding=binding).validate()
    recomputed = cache_identity(binding)
    if recomputed != cache.identity:
        raise TextCacheError("stored text-cache identity does not match its own binding")
    if binding.get("content_identity_sha256") != content_identity(cache.embeddings, cache.recipe_ids):
        raise TextCacheError("stored text-cache embeddings do not match their content identity")
    if expected_identity and cache.identity != expected_identity:
        raise TextCacheError(f"text cache identity {cache.identity} != expected {expected_identity}")
    return cache


# --- PromptHead -------------------------------------------------------------

class PromptHead(nn.Module):
    """`p_prompt = PromptHead(z_r, frozen_recipe_text_embeddings)`.

    One linear projection of the regional embedding into the frozen text space,
    both sides L2-normalized, scaled by the InfoNCE temperature. The text matrix is
    a non-persistent constant buffer: the fast inference path needs no text encoder
    and makes no network call.

    Applicability follows the frozen M9 contract exactly: SYNTHETIC samples only,
    on regions the recipe actually attacked, with valid visibility. Real live and
    real spoof carry no recipe, so the head contributes nothing for them.
    """

    def __init__(self, region_dim: int, text_embeddings: torch.Tensor, *,
                 temperature: float = DEFAULT_PROMPT_TEMPERATURE,
                 regions: int = REGION_COUNT, cache_identity_sha256: str = ""):
        super().__init__()
        if text_embeddings.dim() != 2:
            raise DetectorContractError("frozen recipe text embeddings must be [N_prompt, text_dim]")
        self.region_dim, self.regions = int(region_dim), int(regions)
        self.text_dim = int(text_embeddings.shape[1])
        self.n_prompt = int(text_embeddings.shape[0])
        self.temperature = float(temperature)
        self.cache_identity_sha256 = str(cache_identity_sha256)
        self.project = nn.Linear(self.region_dim, self.text_dim)
        # Frozen constant, not a parameter: a gradient must never reach the text
        # side. `persistent=True` keeps it in the checkpoint so a resumed run
        # cannot silently pick up a different text cache.
        self.register_buffer("text_embeddings",
                             torch.nn.functional.normalize(text_embeddings.float(), dim=-1, eps=1e-6))

    def project_regions(self, region_embeddings: torch.Tensor) -> torch.Tensor:
        """`[B,R,D]` -> L2-normalized `[B,R,text_dim]`, the `z_attack_region` side
        of the InfoNCE term."""
        if region_embeddings.dim() != 3 or region_embeddings.shape[1] != self.regions \
                or region_embeddings.shape[2] != self.region_dim:
            raise DetectorContractError(
                f"PromptHead expects [B,{self.regions},{self.region_dim}], got {tuple(region_embeddings.shape)}")
        return torch.nn.functional.normalize(self.project(region_embeddings), dim=-1, eps=1e-6)

    @staticmethod
    def applicability(is_synthetic: torch.Tensor, attack_region_mask: torch.Tensor | None,
                      region_valid: torch.Tensor) -> torch.Tensor:
        """`[B,R]` bool: synthetic AND attacked region AND valid visibility."""
        if attack_region_mask is None:
            return torch.zeros_like(region_valid, dtype=torch.bool)
        return (is_synthetic.bool().unsqueeze(1) & attack_region_mask.bool() & region_valid.bool())

    def forward(self, region_embeddings: torch.Tensor, applicable: torch.Tensor, *,
                text_embeddings: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Returns the per-region logits, the Table 53 `[B,N_prompt]` reduction, the
        bounded fusion evidence and the projected embeddings the loss needs.

        `text_embeddings` overrides the cached constant. It exists for the
        `prompt: adapter` variant, whose adapter transforms the CACHED matrix rather
        than fine-tuning any text encoder; `frozen_prompt` passes nothing and the
        cached buffer is used unchanged.
        """
        projected = self.project_regions(region_embeddings)
        keep = applicable.bool()
        if keep.shape != (projected.shape[0], self.regions):
            raise DetectorContractError("PromptHead applicability mask must be [B,R]")
        source = self.text_embeddings if text_embeddings is None else text_embeddings
        if source.shape != self.text_embeddings.shape:
            raise DetectorContractError(
                f"prompt text matrix must stay [{self.n_prompt},{self.text_dim}], got {tuple(source.shape)}")
        text = source.to(projected.dtype)
        region_logits = projected @ text.transpose(0, 1) / self.temperature     # [B,R,N_prompt]
        mask = keep.unsqueeze(-1)
        # SPEC_UNDERSPECIFIED reduction (docs/M9_DETECTOR_CONTRACT.md section 4):
        # the max over APPLICABLE regions, and an exact zero when none applies, so
        # a real sample never contributes a spurious prompt logit.
        very_negative = torch.finfo(region_logits.dtype).min
        reduced = region_logits.masked_fill(~mask, very_negative).max(dim=1).values
        any_applicable = keep.any(dim=1)
        prompt_logits = torch.where(any_applicable.unsqueeze(-1), reduced, torch.zeros_like(reduced))
        # `p_prompt_spoof`: max_n softmax(prompt_logits)_n over the attacked
        # regions, 0 when no region is applicable. Bounded in [0,1] as `s_final`
        # requires.
        per_region = region_logits.softmax(dim=-1).max(dim=-1).values                 # [B,R]
        masked = per_region.masked_fill(~keep, 0.0)
        p_prompt_spoof = torch.where(any_applicable, masked.max(dim=1).values,
                                     torch.zeros_like(masked[:, 0]))
        return {"prompt_logits": prompt_logits, "region_prompt_logits": region_logits,
                "p_prompt_spoof": p_prompt_spoof.clamp(0.0, 1.0),
                "prompt_region_embeddings": projected,
                "prompt_applicable": keep.to(projected.dtype)}

    def extra_repr(self) -> str:
        return (f"region_dim={self.region_dim}, text_dim={self.text_dim}, n_prompt={self.n_prompt}, "
                f"temperature={self.temperature}")


def heads_identity(payload: dict[str, Any]) -> str:
    body = {"schema_version": HEADS_SCHEMA_VERSION, **payload}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()
