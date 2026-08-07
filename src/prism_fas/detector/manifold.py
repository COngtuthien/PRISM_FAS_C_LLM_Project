"""M9 multi-prototype regional real manifolds.

Spec section 9.3 (Table 33):

    M_r_real      = { N(mu_rk, Sigma_rk) } for k = 1..K_r
    d_r(x)        = min_k (z_r - mu_rk)^T Sigma_rk^{-1} (z_r - mu_rk)
    assignment_rk = softmax(-d_rk / tau_prototype)

with `K_r = 4` for every region in MVP, diagonal covariance plus an epsilon floor,
K-means initialization on `source_train` LIVE embeddings after detector warm-up, and
prototype update ONLY from live samples with valid region visibility.

Because the spec fixes a diagonal covariance, the Mahalanobis form is exactly
`sum_d (z_d - mu_d)^2 / max(var_d, eps)`. That is the spec's own formula under the
spec's own constraint, not a substitution.

SPEC_UNDERSPECIFIED — distance scale convention. The spec fixes neither the region
embedding dimension `D` nor the scale convention of `d_r`, but section 10.1 fixes
`margin_out = 3.0` and `clean_cap = 3.0` against that same `d_r`. A raw
`D`-dimensional squared Mahalanobis has expectation `D` in-distribution, so at
`D = 256` every live distance is ~250 and BOTH declared constants become
inoperative: `max(0, 3 - d)` is always 0 and `min(d, 3)` is always 3. M9 therefore
uses the per-dimension mean form,

    d_rk = (1/D) * sum_d (z_rd - mu_rkd)^2 / max(var_rkd, eps)

which has expectation 1 in-distribution and puts the spec's own 3.0 exactly where
its numbers imply (a 3x in-distribution deviation). The formula's structure and
every declared weight are unchanged; only the free scale is fixed. This is an
engineering default, recorded in DECISIONS.md and in
`configs/models/m9_detector.yaml`, and is never presented as a spec value.

Nothing here may be updated by a spoof sample, by a synthetic sample or by an
invalid-visibility region; each is enforced and separately tested.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch import nn
from .contracts import REGION_COUNT, REGION_ORDER, DetectorContractError

MANIFOLD_SCHEMA_VERSION = "m9-real-manifold-v1"
DEFAULT_K = 4
DEFAULT_COVARIANCE_EPSILON = 1.0e-4
DEFAULT_TAU_PROTOTYPE = 1.0
DEFAULT_UPDATE_DECAY = 0.99
KMEANS_MAX_ITERATIONS = 100
# SPEC_UNDERSPECIFIED: see the module docstring. "mean_per_dimension" divides the
# diagonal Mahalanobis sum by D; "sum" is the raw form and leaves the spec's
# margin/clean_cap constants inoperative at D = 256.
DISTANCE_SCALE_CONVENTIONS = ("mean_per_dimension", "sum")
DEFAULT_DISTANCE_SCALE_CONVENTION = "mean_per_dimension"


class ManifoldError(RuntimeError):
    """The regional real manifold cannot be built or updated as declared."""


def deterministic_kmeans(points: np.ndarray, k: int, *, seed: int,
                         max_iterations: int = KMEANS_MAX_ITERATIONS) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with seeded k-means++ init and a stable tie-break.

    Determinism is a requirement, not a nicety: the prototype identity is bound into
    the checkpoint, so the same embeddings and seed must give the same centers.
    """
    data = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if data.ndim != 2: raise ManifoldError("k-means expects a [N,D] matrix")
    count = data.shape[0]
    if count < k: raise ManifoldError(f"k-means needs at least k={k} points, got {count}")
    rng = np.random.Generator(np.random.PCG64(int(seed) % (2 ** 32)))
    centers = [data[int(rng.integers(count))]]
    for _ in range(1, k):
        distances = np.min(np.stack([((data - centre) ** 2).sum(axis=1) for centre in centers]), axis=0)
        total = float(distances.sum())
        if total <= 0.0:
            # Degenerate: every point already coincides with a centre. Take the
            # lowest unused index rather than an arbitrary random one.
            chosen = int(np.argmax(distances)) if distances.size else 0
        else:
            chosen = int(np.searchsorted(np.cumsum(distances / total), float(rng.random()), side="right"))
            chosen = min(chosen, count - 1)
        centers.append(data[chosen])
    centre_matrix = np.stack(centers)
    assignment = np.zeros(count, dtype=np.int64)
    for _ in range(int(max_iterations)):
        distances = ((data[:, None, :] - centre_matrix[None, :, :]) ** 2).sum(axis=2)
        # argmin already breaks ties toward the lowest index, which is stable.
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, assignment): break
        assignment = updated
        for index in range(k):
            members = data[assignment == index]
            if members.size: centre_matrix[index] = members.mean(axis=0)
    return centre_matrix.astype(np.float64), assignment


@dataclass(frozen=True)
class PrototypeState:
    """The exact per-region state spec section 9.3 requires."""
    centers: np.ndarray            # [R,K,D]
    variances: np.ndarray          # [R,K,D] diagonal covariance
    counts: np.ndarray             # [R,K] int64
    valid: np.ndarray              # [R,K] bool
    epsilon: float
    region_names: tuple[str, ...] = REGION_ORDER

    def validate(self) -> "PrototypeState":
        regions, k, dim = self.centers.shape
        if regions != REGION_COUNT: raise ManifoldError(f"expected {REGION_COUNT} regions, got {regions}")
        if self.variances.shape != (regions, k, dim): raise ManifoldError("variance shape must match centers")
        if self.counts.shape != (regions, k) or self.valid.shape != (regions, k):
            raise ManifoldError("counts/valid must be [R,K]")
        if not np.isfinite(self.centers).all() or not np.isfinite(self.variances).all():
            raise ManifoldError("prototype state contains non-finite values")
        # Relative tolerance, not absolute: the module holds the buffers in float32
        # and exports float64, so a value clamped at exactly epsilon comes back a
        # few ULPs low. That is a representation artifact, not a floor violation.
        if float(self.variances.min()) < self.epsilon * (1.0 - 1e-5) - 1e-15:
            raise ManifoldError("a variance fell below the declared epsilon floor")
        if tuple(self.region_names) != REGION_ORDER: raise ManifoldError("region ordering drifted")
        return self

    @property
    def k(self) -> int: return int(self.centers.shape[1])
    @property
    def dim(self) -> int: return int(self.centers.shape[2])

    def identity(self, *, config_hash: str, population_identity: str) -> str:
        """Content identity over the logical state. No paths, no machine metadata."""
        payload = {
            "schema_version": MANIFOLD_SCHEMA_VERSION, "region_names": list(self.region_names),
            "k": self.k, "dim": self.dim, "epsilon": float(self.epsilon),
            "centers": np.round(self.centers.astype(np.float64), 10).tolist(),
            "variances": np.round(self.variances.astype(np.float64), 10).tolist(),
            "counts": self.counts.astype(np.int64).tolist(), "valid": self.valid.astype(bool).tolist(),
            "config_hash": config_hash, "population_identity": population_identity}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class RealManifold(nn.Module):
    """Nine regional multi-prototype real manifolds.

    Prototypes live in buffers, not parameters: the spec updates them from live
    embeddings, not by backpropagation, so a gradient must never reach them.
    """

    def __init__(self, dim: int, *, k: int = DEFAULT_K, epsilon: float = DEFAULT_COVARIANCE_EPSILON,
                 tau_prototype: float = DEFAULT_TAU_PROTOTYPE, decay: float = DEFAULT_UPDATE_DECAY,
                 regions: int = REGION_COUNT,
                 distance_scale_convention: str = DEFAULT_DISTANCE_SCALE_CONVENTION):
        super().__init__()
        self.dim, self.k, self.regions = int(dim), int(k), int(regions)
        self.epsilon, self.tau_prototype, self.decay = float(epsilon), float(tau_prototype), float(decay)
        if distance_scale_convention not in DISTANCE_SCALE_CONVENTIONS:
            raise ManifoldError(f"unknown distance scale convention {distance_scale_convention!r}")
        self.distance_scale_convention = str(distance_scale_convention)
        self.register_buffer("centers", torch.zeros(regions, k, dim))
        self.register_buffer("variances", torch.full((regions, k, dim), float(epsilon)))
        self.register_buffer("counts", torch.zeros(regions, k, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(regions, k, dtype=torch.bool))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    # --- distance -----------------------------------------------------------
    def distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """`d_r(x) = min_k (z_r-mu_rk)^T Sigma_rk^{-1} (z_r-mu_rk)`, diagonal.

        Returns `[B,R]`. Before initialization every distance is 0, so an
        uninitialized manifold contributes no evidence rather than noise.
        """
        return self.per_prototype_distance(embeddings).min(dim=-1).values

    def per_prototype_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """`[B,R,K]` squared Mahalanobis distances under the diagonal covariance."""
        if embeddings.dim() != 3 or embeddings.shape[1] != self.regions or embeddings.shape[2] != self.dim:
            raise DetectorContractError(f"region embeddings must be [B,{self.regions},{self.dim}]")
        if not bool(self.initialized):
            return embeddings.new_zeros(embeddings.shape[0], self.regions, self.k)
        delta = embeddings.unsqueeze(2) - self.centers.unsqueeze(0)
        floored = self.variances.clamp_min(self.epsilon).unsqueeze(0)
        distances = (delta * delta / floored).sum(dim=-1)
        if self.distance_scale_convention == "mean_per_dimension":
            distances = distances / float(self.dim)
        # An unfilled prototype slot must never win the min.
        return distances.masked_fill(~self.valid.unsqueeze(0), float("inf")).clamp_min(0.0)

    def soft_min(self, embeddings: torch.Tensor) -> torch.Tensor:
        """`SoftMin_k d_rk` for `L_real`, using the declared `tau_prototype`."""
        distances = self.per_prototype_distance(embeddings)
        if not bool(self.initialized): return distances.sum(dim=-1) * 0.0
        weights = torch.softmax(-distances / self.tau_prototype, dim=-1)
        return (weights * distances.masked_fill(~self.valid.unsqueeze(0), 0.0)).sum(dim=-1)

    def assignment(self, embeddings: torch.Tensor) -> torch.Tensor:
        """`assignment_rk = softmax(-d_rk / tau_prototype)`."""
        return torch.softmax(-self.per_prototype_distance(embeddings) / self.tau_prototype, dim=-1)

    # --- state --------------------------------------------------------------
    def load_state(self, state: PrototypeState) -> None:
        state.validate()
        if state.dim != self.dim or state.k != self.k:
            raise ManifoldError(f"prototype state is [K={state.k},D={state.dim}], module is [K={self.k},D={self.dim}]")
        device = self.centers.device
        self.centers.copy_(torch.as_tensor(state.centers, dtype=torch.float32, device=device))
        self.variances.copy_(torch.as_tensor(state.variances, dtype=torch.float32, device=device).clamp_min(self.epsilon))
        self.counts.copy_(torch.as_tensor(state.counts, dtype=torch.long, device=device))
        self.valid.copy_(torch.as_tensor(state.valid, dtype=torch.bool, device=device))
        self.initialized.fill_(True)

    def export_state(self) -> PrototypeState:
        return PrototypeState(centers=self.centers.detach().cpu().numpy().astype(np.float64),
                              variances=self.variances.detach().cpu().numpy().astype(np.float64),
                              counts=self.counts.detach().cpu().numpy().astype(np.int64),
                              valid=self.valid.detach().cpu().numpy().astype(bool),
                              epsilon=self.epsilon).validate()

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, *, is_live: torch.Tensor, is_synthetic: torch.Tensor,
               region_valid: torch.Tensor) -> dict[str, int]:
        """EMA update from LIVE, non-synthetic samples on VALID regions only.

        Spec section 9.3: "Prototype update chỉ dùng live samples và region
        visibility hợp lệ." A spoof sample, a synthetic sample or an invalid region
        can never move a prototype, and this runs under `no_grad` so no gradient
        reaches the manifold either.
        """
        if not bool(self.initialized): return {"updated": 0, "skipped_not_live": 0}
        usable = is_live.bool() & ~is_synthetic.bool()
        if not usable.any():
            return {"updated": 0, "skipped_not_live": int((~usable).sum())}
        selected = embeddings[usable].detach().float()
        selected_valid = region_valid[usable].bool()
        distances = self.per_prototype_distance(selected)
        nearest = distances.argmin(dim=-1)
        updated = 0
        for region in range(self.regions):
            for index in range(self.k):
                if not bool(self.valid[region, index]): continue
                member = selected_valid[:, region] & (nearest[:, region] == index)
                if not member.any(): continue
                points = selected[member, region, :]
                mean = points.mean(dim=0)
                self.centers[region, index] = self.decay * self.centers[region, index] + (1 - self.decay) * mean
                spread = ((points - self.centers[region, index]) ** 2).mean(dim=0)
                self.variances[region, index] = (self.decay * self.variances[region, index]
                                                 + (1 - self.decay) * spread).clamp_min(self.epsilon)
                self.counts[region, index] += int(member.sum())
                updated += int(member.sum())
        return {"updated": updated, "skipped_not_live": int((~usable).sum())}


def initialize_prototypes(embeddings: np.ndarray, valid: np.ndarray, *, k: int = DEFAULT_K,
                          epsilon: float = DEFAULT_COVARIANCE_EPSILON, seed: int = 20260806) -> PrototypeState:
    """Deterministic K-means initialization on `source_train` LIVE embeddings.

    `embeddings` is `[N,R,D]` and `valid` is `[N,R]`. A region with fewer than `k`
    valid samples raises: the spec fixes K per region for MVP, so silently reducing
    it would change the declared model without saying so.
    """
    data = np.asarray(embeddings, dtype=np.float64)
    mask = np.asarray(valid).astype(bool)
    if data.ndim != 3 or data.shape[1] != REGION_COUNT:
        raise ManifoldError(f"embeddings must be [N,{REGION_COUNT},D]")
    if mask.shape != data.shape[:2]: raise ManifoldError("valid mask must be [N,R]")
    regions, dim = REGION_COUNT, data.shape[2]
    centers = np.zeros((regions, k, dim), dtype=np.float64)
    variances = np.full((regions, k, dim), float(epsilon), dtype=np.float64)
    counts = np.zeros((regions, k), dtype=np.int64)
    flags = np.zeros((regions, k), dtype=bool)
    shortfalls: list[str] = []
    for region in range(regions):
        points = data[mask[:, region], region, :]
        if points.shape[0] < k:
            shortfalls.append(f"{REGION_ORDER[region]}={points.shape[0]}")
            continue
        # Deterministic ordering before clustering: the caller's row order must not
        # decide the result.
        order = np.lexsort(points.T[::-1])
        centre_matrix, assignment = deterministic_kmeans(points[order], k, seed=seed + region)
        for index in range(k):
            members = points[order][assignment == index]
            centers[region, index] = centre_matrix[index]
            counts[region, index] = int(members.shape[0])
            flags[region, index] = members.shape[0] > 0
            if members.shape[0] >= 2:
                variances[region, index] = np.maximum(members.var(axis=0), epsilon)
    if shortfalls:
        raise ManifoldError(f"regions with fewer than k={k} valid live samples: {sorted(shortfalls)}")
    return PrototypeState(centers=centers, variances=variances, counts=counts, valid=flags,
                          epsilon=float(epsilon)).validate()


def write_prototypes_npz(path: Any, state: PrototypeState, *, config_hash: str,
                         population_identity: str, feature_identity: str) -> dict[str, Any]:
    """Export `prototypes.npz`. Arrays only, so it loads with `allow_pickle=False`."""
    from pathlib import Path
    import io, zipfile
    identity = state.identity(config_hash=config_hash, population_identity=population_identity)
    arrays = {
        "region_names": np.array(list(state.region_names), dtype="U16"),
        "k": np.array([state.k], dtype=np.int64), "dim": np.array([state.dim], dtype=np.int64),
        "centers": state.centers.astype(np.float64), "variances": state.variances.astype(np.float64),
        "counts": state.counts.astype(np.int64), "valid": state.valid.astype(bool),
        "epsilon": np.array([state.epsilon], dtype=np.float64),
        "config_hash": np.array([config_hash], dtype="U64"),
        "population_identity": np.array([population_identity], dtype="U64"),
        "feature_contract_identity": np.array([feature_identity], dtype="U64"),
        "prototype_identity": np.array([identity], dtype="U64"),
        "schema_version": np.array([MANIFOLD_SCHEMA_VERSION], dtype="U32")}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    # Fixed member dates: a wall clock in the archive would break byte determinism.
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload.getvalue())
    data = buffer.getvalue()
    target.write_bytes(data)
    return {"prototype_identity_sha256": identity, "bytes": len(data),
            "file_sha256": hashlib.sha256(data).hexdigest(), "regions": list(state.region_names),
            "k": state.k, "dim": state.dim, "epsilon": state.epsilon}


def read_prototypes_npz(path: Any) -> tuple[PrototypeState, dict[str, Any]]:
    from pathlib import Path
    with np.load(Path(path), allow_pickle=False) as handle:
        state = PrototypeState(centers=handle["centers"], variances=handle["variances"],
                               counts=handle["counts"], valid=handle["valid"],
                               epsilon=float(handle["epsilon"][0]),
                               region_names=tuple(str(name) for name in handle["region_names"]))
        meta = {"prototype_identity_sha256": str(handle["prototype_identity"][0]),
                "config_hash": str(handle["config_hash"][0]),
                "population_identity": str(handle["population_identity"][0]),
                "feature_contract_identity": str(handle["feature_contract_identity"][0]),
                "schema_version": str(handle["schema_version"][0])}
    return state.validate(), meta
