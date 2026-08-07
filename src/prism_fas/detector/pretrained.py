"""Pinned pretrained backbones for M9, and their SHA-verified resolution.

Spec section 1.2 (Table 5) fixes the two branches:

    local  = ConvNeXt V2 Atto or Tiny
    global = SigLIP2 Base P16-224, FROZEN, train fusion/heads

Spec section 18.1 (Table 69) fixes what a pinned model must record: component,
model id, revision, file SHA-256, license, local relative path, input contract and
output contract. Every field below was read from the real downloaded bytes — no
revision is `main` or `latest`, and the SHA-256 is the actual pin.

Nothing here may import modal. Weights are never vendored into git.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRETRAINED_SCHEMA_VERSION = "m9-pretrained-v1"

# The exact revision resolved from the Hub, not a branch name. `model.safetensors`
# holds both towers; M9 freezes the whole SigLIP2 model and trains only fusion and
# heads on top of it.
SIGLIP2_PIN: dict[str, Any] = {
    "component": "siglip",
    "model_id": "google/siglip2-base-patch16-224",
    "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "license": "Apache-2.0",
    "local_relpath": "pretrained/m9/siglip2",
    "architecture": "SiglipModel",
    "tokenizer_class": "GemmaTokenizer",
    "image_size": 224, "patch_size": 16, "vision_hidden_size": 768,
    "text_hidden_size": 768, "text_max_position_embeddings": 64, "text_vocab_size": 256000,
    # 224/16 = 14 -> 14*14 patch tokens from the vision tower.
    "vision_patch_tokens": 196,
    "input_contract": {"color": "rgb", "size": [224, 224], "value_range": [0.0, 1.0],
                       "normalization_mean": [0.5, 0.5, 0.5], "normalization_std": [0.5, 0.5, 0.5]},
    "output_contract": {"patch_tokens": "[B,196,768] last_hidden_state",
                        "pooled": "[B,768] pooler_output", "text": "[B,768] pooler_output"},
    "notes": "frozen in M9; fusion and heads train on top. Text encoder is used offline "
             "only, to build the cached recipe text embeddings.",
    "files": {
        "config.json": {"sha256": "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7", "bytes": 253},
        "model.safetensors": {"sha256": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b", "bytes": 1500800904},
        "preprocessor_config.json": {"sha256": "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc", "bytes": 394},
        "special_tokens_map.json": {"sha256": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351", "bytes": 636},
        "tokenizer.json": {"sha256": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322", "bytes": 34363039},
        "tokenizer.model": {"sha256": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2", "bytes": 4241003},
        "tokenizer_config.json": {"sha256": "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8", "bytes": 47164}},
    # Files that must exist and hash-match before the model is used at all.
    "required_files": ("config.json", "model.safetensors", "preprocessor_config.json",
                       "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
                       "tokenizer_config.json"),
    # Only these participate in the recipe-text cache identity: the weights, the
    # config that shapes the towers, and everything that decides tokenization.
    "text_identity_files": ("config.json", "model.safetensors", "special_tokens_map.json",
                            "tokenizer.json", "tokenizer.model", "tokenizer_config.json")}

# SPEC_UNDERSPECIFIED: the spec names SigLIP2 but not its tokenization call. The
# model's own `text_max_position_embeddings` is 64, so `max_length=64` is the
# model's contract rather than a free choice; padding/truncation are declared here
# and bound into the cache identity.
SIGLIP2_TOKENIZATION: dict[str, Any] = {
    "padding": "max_length", "max_length": 64, "truncation": True,
    "add_special_tokens": True, "use_fast": False, "lowercase": False}

# Pinned by M5/M6 and already verified and uploaded; reused rather than re-pinned.
CONVNEXT_PIN: dict[str, Any] = {
    "component": "convnext",
    "model_id": "timm/convnextv2_atto.fcmae_ft_in1k",
    "timm_name": "convnextv2_atto.fcmae_ft_in1k",
    "revision": "timm/convnextv2_atto.fcmae_ft_in1k",
    "license": "CC-BY-NC-4.0",
    "local_relpath": "pretrained/m9/convnextv2_atto/model.safetensors",
    # Relative to whatever weight root the caller passes. The first two cover the
    # Modal volume layout (`/vol/models/pretrained/m9`) and the local model cache
    # layout; the SHA-256 is the pin, so the directory shape may differ.
    "alternate_relpaths": ("convnextv2_atto/model.safetensors", "backbones/model.safetensors",
                           "model.safetensors", "pretrained/m9/convnextv2_atto/model.safetensors",
                           "pretrained/m8/backbones/model.safetensors"),
    "weight_sha256": "6389c2f5a427b01a922e66e6d352c707424cccb62390c6936bc612e3d10b7ebb",
    "input_contract": {"color": "rgb", "size": [224, 224], "value_range": [0.0, 1.0],
                       "normalization_mean": [0.485, 0.456, 0.406],
                       "normalization_std": [0.229, 0.224, 0.225]},
    # Stage-4 stride 32 at 224 input -> 7x7 = 49 local tokens (`P` in Table 53).
    "output_contract": {"stage4": "[B,320,7,7]", "local_tokens": 49, "channels": 320},
    "notes": "spec allows Atto or Tiny; Atto is already pinned, verified and uploaded."}


class PretrainedError(RuntimeError):
    """A pinned pretrained artifact is missing or does not match its pinned SHA-256."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SigLIP2Artifacts:
    """A SHA-verified local SigLIP2 directory.

    `root` is a runtime path and never enters an identity; only the digests do.
    """
    root: Path
    digests: dict[str, str]

    @classmethod
    def resolve(cls, weight_root: Path, *, verify: bool = True) -> "SigLIP2Artifacts":
        """Locate the pinned SigLIP2 under either the local cache layout or the
        Modal volume layout and verify every required file byte-for-byte."""
        root = Path(weight_root)
        candidates = [root, root / "siglip2", root / "pretrained" / "m9" / "siglip2", root / "m9" / "siglip2"]
        found = next((path for path in candidates if (path / "config.json").is_file()), None)
        if found is None:
            raise PretrainedError(f"pinned SigLIP2 not found under {root.name}; expected a directory holding "
                                  f"{list(SIGLIP2_PIN['required_files'])}")
        digests: dict[str, str] = {}
        for name in SIGLIP2_PIN["required_files"]:
            path = found / name
            if not path.is_file(): raise PretrainedError(f"pinned SigLIP2 file {name} is missing")
            digest = sha256_file(path)
            expected = SIGLIP2_PIN["files"][name]["sha256"]
            if verify and digest != expected:
                raise PretrainedError(f"SigLIP2 {name} SHA {digest} != pinned {expected}; refusing to substitute a model")
            digests[name] = digest
        return cls(root=found, digests=digests)

    def identity(self) -> str:
        """Hash of the pin plus the verified digests. Path-free and machine-free."""
        payload = {"schema_version": PRETRAINED_SCHEMA_VERSION,
                   "model_id": SIGLIP2_PIN["model_id"], "revision": SIGLIP2_PIN["revision"],
                   "architecture": SIGLIP2_PIN["architecture"],
                   "tokenizer_class": SIGLIP2_PIN["tokenizer_class"],
                   "digests": {name: self.digests[name] for name in sorted(self.digests)},
                   "tokenization": dict(sorted(SIGLIP2_TOKENIZATION.items()))}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, Any]:
        spec = {key: value for key, value in SIGLIP2_PIN.items()
                if key not in ("files", "required_files", "text_identity_files", "local_relpath")}
        return {**spec, "tokenization": dict(SIGLIP2_TOKENIZATION),
                "verified_sha256": dict(sorted(self.digests.items())),
                "sha256_matches_pin": all(self.digests[name] == SIGLIP2_PIN["files"][name]["sha256"]
                                          for name in self.digests),
                "siglip2_identity_sha256": self.identity()}

    def load_model(self, *, device: str = "cpu", dtype: Any = None) -> Any:
        """The frozen SigLIP2 model. Never trains, never downloads.

        `local_files_only=True` is not a convenience: a silent Hub fetch would
        replace a verified pin with whatever is current.
        """
        import torch
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(self.root), local_files_only=True,
                                          dtype=dtype or torch.float32)
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        return model.to(device)

    def load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(self.root), local_files_only=True,
                                             use_fast=bool(SIGLIP2_TOKENIZATION["use_fast"]))


def resolve_convnext_weight(weight_root: Path, *, verify: bool = True) -> Path:
    """The pinned ConvNeXt V2 Atto weight file, wherever the layout puts it."""
    root = Path(weight_root)
    path = next((root / relative for relative in CONVNEXT_PIN["alternate_relpaths"]
                 if (root / relative).is_file()), None)
    if path is None:
        raise PretrainedError(f"pinned ConvNeXt weight missing under {root.name}; "
                              f"looked for {list(CONVNEXT_PIN['alternate_relpaths'])}")
    if verify:
        digest = sha256_file(path)
        if digest != CONVNEXT_PIN["weight_sha256"]:
            raise PretrainedError(f"ConvNeXt weight SHA {digest} != pinned {CONVNEXT_PIN['weight_sha256']}")
    return path


def pretrained_manifest(siglip: SigLIP2Artifacts | None = None,
                        convnext_sha256: str | None = None) -> dict[str, Any]:
    """The Table 69 registry entry set for M9."""
    entry = {key: value for key, value in CONVNEXT_PIN.items() if key != "alternate_relpaths"}
    return {"pretrained_schema_version": PRETRAINED_SCHEMA_VERSION,
            "convnext": {**entry, "verified_sha256": convnext_sha256,
                         "sha256_matches_pin": convnext_sha256 == CONVNEXT_PIN["weight_sha256"]
                         if convnext_sha256 else None},
            "siglip2": siglip.manifest() if siglip is not None else
                       {key: value for key, value in SIGLIP2_PIN.items() if key != "files"}}
