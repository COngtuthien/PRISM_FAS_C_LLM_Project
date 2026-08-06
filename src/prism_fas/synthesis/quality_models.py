from __future__ import annotations
import hashlib, importlib.util, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import yaml

QUALITY_MODELS_SCHEMA_VERSION = "m8-quality-models-v1"
# Pinned in M3 and verified against the real files; never substituted after
# seeing quality results.
PINNED = {
    "detector": {"role": "detection_and_landmarks", "backend": "scrfd_10g_bnkps", "file": "scrfd_10g_bnkps.onnx",
                 "relative_path": "face_detectors/scrfd_10g_bnkps.onnx",
                 "alternate_relative_paths": ("face_detectors/scrfd_10g_bnkps.onnx",), "revision": None,
                 "revision_note": "pinned local ONNX artifact; no upstream revision string, the SHA-256 is the pin",
                 "sha256": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
                 "input_size": 320, "threshold": 0.5, "policy": "scrfd_source_policy_v1"},
    "parsing": {"role": "parsing_and_pose", "backend": "facexformer", "file": "model.pt",
                "relative_path": "face_geometry/ckpts/model.pt",
                "alternate_relative_paths": ("face_geometry/ckpts/model.pt", "face_geometry/model.pt"),
                "repo_id": "kartiknarayan/facexformer",
                "revision": "fd12148d0b19",
                "sha256": "327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d",
                "input_size": 224, "num_classes": 11},
    "identity": {"role": "identity", "backend": "adaface_ir50", "file": "model.pt",
                 "relative_path": "face_identity/pretrained_model/model.pt",
                 "alternate_relative_paths": ("face_identity/pretrained_model/model.pt", "face_identity/model.pt"),
                 "repo_id": "minchul/cvlface_adaface_ir50_webface4m", "revision": "60a65befbcf7",
                 "sha256": "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496",
                 "input_size": 112, "embedding_dim": 512, "input_color": "bgr"}}


class QualityModelError(RuntimeError):
    """A pinned quality model is missing or does not match its pinned SHA-256."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def resolve_weight(weight_root: Path, role: str, *, verify: bool = True) -> Path:
    """Locate a pinned weight under either the local cache layout or the Modal
    volume layout. The SHA-256 is the pin, so the directory shape may differ."""
    spec = PINNED[role]
    candidates = spec.get("alternate_relative_paths") or (spec["relative_path"],)
    path = next((Path(weight_root) / relative for relative in candidates
                 if (Path(weight_root) / relative).is_file()), None)
    if path is None:
        raise QualityModelError(f"pinned {role} weight missing; looked for {list(candidates)} under the weight root")
    if verify:
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise QualityModelError(f"{role} weight SHA {digest} != pinned {spec['sha256']}; refusing to substitute a model")
    return path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise QualityModelError(f"cannot load third-party module from {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DifferentiableAdaFace(torch.nn.Module):
    """Frozen AdaFace IR-50 used as a differentiable feature extractor.

    Parameters never receive gradients (`requires_grad_(False)`, `eval()`), but
    gradient flows *through* the network into the generated image, which is what
    the identity loss needs. Both the generated and the cached live embedding are
    produced by this same wrapper so their cosine is self-consistent — the M3
    package embeddings came from an OpenCV resize path and are not mixed in here.
    """
    def __init__(self, weight_path: Path, code_root: Path, *, architecture: str = "ir_50",
                 input_size: int = 112, embedding_dim: int = 512):
        super().__init__()
        module = _load_module("adaface_net", Path(code_root) / "adaface_net.py")
        self.net = module.build_model(architecture)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = {key[4:]: value for key, value in checkpoint.items() if key.startswith("net.")}
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        for parameter in self.net.parameters(): parameter.requires_grad_(False)
        self.input_size, self.embedding_dim = int(input_size), int(embedding_dim)

    def train(self, mode: bool = True) -> "DifferentiableAdaFace":
        # The frozen extractor stays in eval mode even inside a training loop.
        return super().train(False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: [B,3,H,W] float in [0,1] -> L2-normalized [B,512]."""
        if rgb.dim() != 4 or rgb.shape[1] != 3: raise QualityModelError("AdaFace input must be [B,3,H,W]")
        bgr = rgb.flip(1)
        if bgr.shape[-1] != self.input_size or bgr.shape[-2] != self.input_size:
            bgr = torch.nn.functional.interpolate(bgr, size=(self.input_size, self.input_size),
                                                  mode="bicubic", align_corners=False)
        normalized = (bgr.clamp(0.0, 1.0) - 0.5) / 0.5
        features, _ = self.net(normalized.to(next(self.net.parameters()).dtype))
        if features.shape[1] != self.embedding_dim:
            raise QualityModelError(f"AdaFace returned {features.shape[1]} dims, expected {self.embedding_dim}")
        return torch.nn.functional.normalize(features.float(), dim=1, eps=1e-6)


@dataclass
class QualityModelRegistry:
    """Resolved, SHA-verified pinned model paths plus their recorded provenance."""
    weight_root: Path
    code_root: Path
    verified: dict[str, str]

    @classmethod
    def resolve(cls, weight_root: Path, *, roles: tuple[str, ...] = ("identity",), verify: bool = True) -> "QualityModelRegistry":
        root = Path(weight_root)
        verified = {role: sha256_file(resolve_weight(root, role, verify=verify)) for role in roles}
        return cls(weight_root=root, code_root=root / "code", verified=verified)

    def adaface(self, device: str = "cpu") -> DifferentiableAdaFace:
        spec = PINNED["identity"]
        model = DifferentiableAdaFace(resolve_weight(self.weight_root, "identity"), self.code_root / "adaface",
                                      input_size=int(spec["input_size"]), embedding_dim=int(spec["embedding_dim"]))
        return model.to(device)

    def manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = {"quality_models_schema_version": QUALITY_MODELS_SCHEMA_VERSION, "models": {}}
        for role, digest in sorted(self.verified.items()):
            spec = dict(PINNED[role])
            spec.pop("relative_path", None)
            out["models"][role] = {**spec, "verified_sha256": digest, "sha256_matches_pin": digest == PINNED[role]["sha256"]}
        return out


def load_identity_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(payload.get("identity_model", {}))
