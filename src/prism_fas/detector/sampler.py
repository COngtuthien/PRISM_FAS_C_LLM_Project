"""The deterministic M9 batch sampler. Spec section 10.2 (Table 36, Table 65).

    Real live       37.5 %   12 of 32   CASIA/MSU domain-balanced, required every batch
    Real spoof      37.5 %   12 of 32   CASIA/MSU domain-balanced, required every batch
    Synthetic spoof 25.0 %    8 of 32   quality-weighted, mix physics and GPAT

Determinism is seeded by `base_seed + epoch` (spec section 12.7). Every stream is
derived through SHA-256, never through Python's randomized `hash()`, so the batch
sequence does not depend on `PYTHONHASHSEED` or on the process.

Gradient accumulation keeps the EFFECTIVE composition exact across the accumulation
window (spec section 10.2): the micro-batch shrinks, the ratio does not.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence
import numpy as np

SAMPLER_SCHEMA_VERSION = "m9-batch-sampler-v1"
# Table 65: batch: {live: 12, real_spoof: 12, synthetic_spoof: 8}.
DEFAULT_COMPOSITION = {"real_live": 12, "real_spoof": 12, "synthetic": 8}
DOMAINS = ("casia_fasd", "msu_mfsd")
ROUTES = ("physics", "gpat")
LIVE_KIND, SPOOF_KIND = "live", "spoof"


class SamplerError(ValueError):
    """The requested batch composition cannot be honoured with these pools."""


def stream_seed(*parts: Any) -> int:
    """A stable 64-bit seed from SHA-256 of the joined parts.

    Never `hash()`: that is randomized per process and would make a resumed run
    diverge from the run it resumed.
    """
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def generator(*parts: Any) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream_seed(*parts)))


@dataclass(frozen=True)
class BatchContract:
    """The frozen composition, hashed into every checkpoint."""
    real_live: int = DEFAULT_COMPOSITION["real_live"]
    real_spoof: int = DEFAULT_COMPOSITION["real_spoof"]
    synthetic: int = DEFAULT_COMPOSITION["synthetic"]
    domain_balance: bool = True
    require_both_routes: bool = True
    accumulation_steps: int = 1
    # G1/G2 read "Source real live/spoof" (Table 39): the synthetic bank does not
    # exist for them, so a real-only warm-up contract is explicitly allowed. It is
    # never allowed for G5, whose input is "Real + synthetic".
    phase: str = "mixed"
    # Which routes of the accepted M8 bank this row is allowed to draw from. The
    # A03 single-route variants restrict the ACCEPTED rows; they never regenerate a
    # bank and never change a quality threshold.
    routes: tuple[str, ...] = ROUTES
    # Which SOURCE domains the real partitions are drawn from. §19: P1 trains on
    # CASIA alone and P2 on MSU alone, so "domain-balanced" for those protocols
    # means balanced over the one domain they have. Defaults to both, which is
    # what every Version-B row used, so the reference contract identity does not
    # move.
    domains: tuple[str, ...] = DOMAINS

    @property
    def batch_size(self) -> int: return self.real_live + self.real_spoof + self.synthetic

    def validate(self) -> "BatchContract":
        if self.phase not in ("mixed", "real_only"):
            raise SamplerError(f"unknown batch phase {self.phase!r}")
        if min(self.real_live, self.real_spoof) <= 0:
            raise SamplerError("every batch must contain real live AND real spoof (Table 9)")
        if not self.domain_balance and int(self.accumulation_steps) > 1:
            raise SamplerError("naive concatenation draws a variable per-class split, which cannot be "
                               "guaranteed to divide into accumulation micro-batches; declare "
                               "accumulation_steps=1 for this sampler")
        if self.phase == "real_only":
            if self.synthetic: raise SamplerError("a real-only warm-up batch cannot carry synthetic samples")
        elif self.synthetic <= 0:
            raise SamplerError("mixed training requires synthetic samples in every batch (Table 36)")
        if self.synthetic * 4 > self.batch_size:
            raise SamplerError("synthetic may never exceed 25 % of a batch (Table 9)")
        if not self.domains:
            raise SamplerError("a batch contract must declare at least one source domain")
        unknown_domains = sorted(set(self.domains) - set(DOMAINS))
        if unknown_domains: raise SamplerError(f"unknown source domain(s) {unknown_domains}")
        if self.domain_balance and (self.real_live % len(self.domains)
                                    or self.real_spoof % len(self.domains)):
            raise SamplerError(
                f"domain-balanced real quotas must divide by {len(self.domains)} "
                f"for domains {list(self.domains)}")
        if self.synthetic and not self.routes:
            raise SamplerError("a mixed batch must declare at least one synthetic route")
        unknown = sorted(set(self.routes) - set(ROUTES))
        if unknown: raise SamplerError(f"unknown synthetic route(s) {unknown}")
        if self.require_both_routes and len(self.routes) < len(ROUTES):
            raise SamplerError("a single-route bank cannot be asked to carry both routes in every batch")
        steps = int(self.accumulation_steps)
        if steps < 1: raise SamplerError("accumulation_steps must be >= 1")
        if steps > 1 and any(count % steps for count in (self.real_live, self.real_spoof, self.synthetic)):
            raise SamplerError(f"accumulation_steps={steps} cannot preserve the exact effective composition")
        return self

    def payload(self) -> dict[str, Any]:
        body = {"schema_version": SAMPLER_SCHEMA_VERSION, "real_live": self.real_live,
                "real_spoof": self.real_spoof, "synthetic": self.synthetic,
                "batch_size": self.batch_size, "domain_balance": self.domain_balance,
                "require_both_routes": self.require_both_routes and bool(self.synthetic),
                "accumulation_steps": self.accumulation_steps, "phase": self.phase,
                "ratios": {"real_live": self.real_live / self.batch_size,
                           "real_spoof": self.real_spoof / self.batch_size,
                           "synthetic": self.synthetic / self.batch_size}}
        # Carried only when the row restricts the accepted bank to one route, so the
        # reference contract keeps the hash its M9 checkpoints were written with.
        if tuple(self.routes) != ROUTES: body["routes"] = list(self.routes)
        # Same rule for the protocol's source domains: present only when the row
        # is not the both-domains default, so a P1/P2 contract hashes differently
        # from a P3-ready one and neither disturbs the inherited reference hash.
        if tuple(self.domains) != DOMAINS: body["domains"] = list(self.domains)
        return body

    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BatchPlan:
    """One batch as index lists into the real and synthetic pools."""
    step: int
    epoch: int
    real_live: tuple[int, ...]
    real_spoof: tuple[int, ...]
    synthetic: tuple[int, ...]

    @property
    def size(self) -> int: return len(self.real_live) + len(self.real_spoof) + len(self.synthetic)

    def microbatches(self, accumulation_steps: int) -> list["BatchPlan"]:
        """Split into `accumulation_steps` micro-batches that each carry the exact
        declared ratio, so the effective composition over the window is unchanged."""
        steps = int(accumulation_steps)
        if steps < 1: raise SamplerError("accumulation_steps must be >= 1")
        if steps == 1: return [self]
        for name, values in (("real_live", self.real_live), ("real_spoof", self.real_spoof),
                             ("synthetic", self.synthetic)):
            if len(values) % steps: raise SamplerError(f"{name} ({len(values)}) does not split into {steps} micro-batches")
        chunk = lambda values, index: tuple(values[index * (len(values) // steps):(index + 1) * (len(values) // steps)])
        return [BatchPlan(step=self.step, epoch=self.epoch, real_live=chunk(self.real_live, index),
                          real_spoof=chunk(self.real_spoof, index), synthetic=chunk(self.synthetic, index))
                for index in range(steps)]


class _Stream:
    """A deterministic, endlessly cycling permutation of one pool.

    Reshuffled with a fresh derived seed on every cycle, so a minority pool repeats
    without ever repeating the same order.
    """

    def __init__(self, members: Sequence[int], *, seed_parts: tuple[Any, ...]):
        self.members = list(members)
        self.seed_parts = seed_parts
        self.cycle = 0
        self.order: list[int] = []
        self.cursor = 0
        if not self.members: raise SamplerError(f"pool {seed_parts!r} is empty")
        self._reshuffle()

    def _reshuffle(self) -> None:
        rng = generator(*self.seed_parts, self.cycle)
        self.order = [self.members[index] for index in rng.permutation(len(self.members))]
        self.cursor = 0
        self.cycle += 1

    def take(self, count: int) -> list[int]:
        out: list[int] = []
        while len(out) < count:
            if self.cursor >= len(self.order): self._reshuffle()
            out.append(self.order[self.cursor])
            self.cursor += 1
        return out


class M9BatchSampler:
    """Deterministic 12/12/8 batch plans over the real and synthetic pools.

    The sequence is a pure function of `(seed, epoch, pool identities)`. Resume
    replays the same epoch and skips to the recorded step, so the next batch after
    a resume is exactly the batch the interrupted run would have produced.
    """

    def __init__(self, *, real_live: dict[str, Sequence[int]], real_spoof: dict[str, Sequence[int]],
                 synthetic_routes: dict[str, Sequence[int]], contract: BatchContract, seed: int,
                 steps_per_epoch: int, identity: str = ""):
        self.contract = contract.validate()
        self.seed = int(seed)
        self.steps_per_epoch = int(steps_per_epoch)
        self.identity = str(identity)
        if self.steps_per_epoch <= 0: raise SamplerError("steps_per_epoch must be positive")
        self.real_live_pools = {name: list(values) for name, values in sorted(real_live.items())}
        self.real_spoof_pools = {name: list(values) for name, values in sorted(real_spoof.items())}
        # A row that declares one route sees ONLY that route's accepted rows. The
        # bank is not rebuilt and no threshold moves; the pool is restricted.
        self.synthetic_pools = {name: list(values) for name, values in sorted(synthetic_routes.items())
                                if name in self.contract.routes}
        if self.contract.domain_balance:
            for label, pools in (("live", self.real_live_pools), ("spoof", self.real_spoof_pools)):
                missing = [name for name in self.contract.domains if not pools.get(name)]
                if missing: raise SamplerError(f"domain balance needs real {label} samples from {missing}")
        else:
            for label, pools in (("live", self.real_live_pools), ("spoof", self.real_spoof_pools)):
                if not any(pools.values()):
                    raise SamplerError(f"naive concatenation still needs at least one real {label} row (Table 9)")
        if self.contract.synthetic:
            missing = [name for name in self.contract.routes if not self.synthetic_pools.get(name)]
            if missing: raise SamplerError(f"the synthetic pool is missing declared route(s) {missing}")

    # --- state --------------------------------------------------------------
    def state(self, *, epoch: int, step: int) -> dict[str, Any]:
        return {"schema_version": SAMPLER_SCHEMA_VERSION, "seed": self.seed, "epoch": int(epoch),
                "step": int(step), "steps_per_epoch": self.steps_per_epoch,
                "batch_contract_identity": self.contract.identity(),
                "batch_contract": self.contract.payload(), "pool_identity": self.identity,
                "pool_sizes": {"real_live": {name: len(values) for name, values in self.real_live_pools.items()},
                               "real_spoof": {name: len(values) for name, values in self.real_spoof_pools.items()},
                               "synthetic": {name: len(values) for name, values in self.synthetic_pools.items()}}}

    # --- one epoch ----------------------------------------------------------
    def _streams(self, epoch: int) -> dict[str, _Stream]:
        base = (SAMPLER_SCHEMA_VERSION, self.identity, self.seed, int(epoch))
        streams: dict[str, _Stream] = {}
        for name, members in self.real_live_pools.items():
            streams[f"live/{name}"] = _Stream(members, seed_parts=(*base, "real_live", name))
        for name, members in self.real_spoof_pools.items():
            streams[f"spoof/{name}"] = _Stream(members, seed_parts=(*base, "real_spoof", name))
        if self.contract.synthetic:
            for name, members in self.synthetic_pools.items():
                streams[f"synthetic/{name}"] = _Stream(members, seed_parts=(*base, "synthetic", name))
        return streams

    def _real(self, streams: dict[str, _Stream], prefix: str, quota: int) -> list[int]:
        # Table 36 requires domain balance on both real partitions — over the
        # domains the PROTOCOL declares, which for P1/P2 is a single one.
        domains = self.contract.domains
        per_domain = quota // len(domains)
        out: list[int] = []
        for name in domains: out.extend(streams[f"{prefix}/{name}"].take(per_domain))
        return out

    def _naive_real(self, epoch: int, step: int) -> tuple[list[int], list[int]]:
        """The A01 `naive_concat` control: draw the real quota from a NAIVE
        CONCATENATION of every real `source_train` row.

        Table 60 asks "naive concat vs domain/class balanced", so this drops BOTH
        balances: no per-domain quota and no per-class quota. Whatever proportion of
        CASIA/MSU and live/spoof the concatenated pool happens to have is the
        proportion the batch gets, which is the entire point of hypothesis H1.

        The ONE thing it does not drop is Table 9's hard requirement that every batch
        contains real live AND real spoof. When a draw is single-class the last slot
        is deterministically replaced by the next draw from the absent class — a
        presence repair, never a ratio repair, and exactly the pattern the synthetic
        route mixing already uses.
        """
        quota = self.contract.real_live + self.contract.real_spoof
        pooled = [(LIVE_KIND, index) for values in self.real_live_pools.values() for index in values]
        pooled += [(SPOOF_KIND, index) for values in self.real_spoof_pools.values() for index in values]
        pooled.sort()
        if len(pooled) < quota:
            raise SamplerError(f"the concatenated real pool holds {len(pooled)} rows, quota is {quota}")
        rng = generator(SAMPLER_SCHEMA_VERSION, self.identity, self.seed, int(epoch),
                        "naive_concat_real", int(step))
        chosen = [pooled[index] for index in rng.choice(len(pooled), size=quota, replace=False)]
        present = {kind for kind, _ in chosen}
        for kind, bucket in ((LIVE_KIND, self.real_live_pools), (SPOOF_KIND, self.real_spoof_pools)):
            if kind in present: continue
            candidates = sorted(index for values in bucket.values() for index in values)
            if not candidates: raise SamplerError(f"the real pool has no {kind} rows at all")
            repair = generator(SAMPLER_SCHEMA_VERSION, self.identity, self.seed, int(epoch),
                               "naive_concat_repair", kind, int(step))
            chosen[-1] = (kind, candidates[int(repair.integers(len(candidates)))])
            present.add(kind)
        return ([index for kind, index in chosen if kind == LIVE_KIND],
                [index for kind, index in chosen if kind == SPOOF_KIND])

    def _synthetic(self, streams: dict[str, _Stream], quota: int, *, epoch: int, step: int) -> list[int]:
        """`quota` draws from the pooled accepted set.

        The spec asks for a MIX of physics and GPAT but fixes no ratio, so no ratio
        is imposed: the draw is over the union, and route presence is repaired
        deterministically only when a batch would otherwise be single-route.
        """
        if int(quota) <= 0: return []
        allowed = tuple(name for name in ROUTES if name in self.contract.routes)
        route_streams = [streams[f"synthetic/{name}"] for name in allowed if f"synthetic/{name}" in streams]
        pooled: list[tuple[str, int]] = []
        for name in allowed:
            key = f"synthetic/{name}"
            if key in streams: pooled.extend((name, index) for index in streams[key].members)
        if not pooled: raise SamplerError(f"no accepted synthetic rows for route(s) {list(allowed)}")
        rng = generator(SAMPLER_SCHEMA_VERSION, self.identity, self.seed, int(epoch), "synthetic_batch", int(step))
        chosen = [pooled[index] for index in rng.choice(len(pooled), size=int(quota), replace=False)]
        if self.contract.require_both_routes and len({route for route, _ in chosen}) < len(route_streams):
            present = {route for route, _ in chosen}
            for name in allowed:
                if name in present or f"synthetic/{name}" not in streams: continue
                # Deterministic repair: the last slot is replaced by the next draw
                # from the absent route's own stream.
                chosen[-1] = (name, streams[f"synthetic/{name}"].take(1)[0])
                present.add(name)
        return [index for _, index in chosen]

    def epoch_plans(self, epoch: int) -> list[BatchPlan]:
        streams = self._streams(epoch)
        plans: list[BatchPlan] = []
        for step in range(self.steps_per_epoch):
            if self.contract.domain_balance:
                live = self._real(streams, "live", self.contract.real_live)
                spoof = self._real(streams, "spoof", self.contract.real_spoof)
            else:
                live, spoof = self._naive_real(epoch, step)
            plan = BatchPlan(step=step, epoch=int(epoch), real_live=tuple(live), real_spoof=tuple(spoof),
                             synthetic=tuple(self._synthetic(streams, self.contract.synthetic,
                                                             epoch=epoch, step=step)))
            if plan.size != self.contract.batch_size:
                raise SamplerError(f"batch {step} has {plan.size} items, expected {self.contract.batch_size}")
            plans.append(plan)
        return plans

    def __iter__(self) -> Iterator[BatchPlan]:
        return iter(self.epoch_plans(0))

    def iter_epoch(self, epoch: int, *, start_step: int = 0) -> Iterator[BatchPlan]:
        """Replay `epoch` from `start_step`. Resume uses this and continues the
        exact sequence rather than restarting the epoch."""
        for plan in self.epoch_plans(epoch)[int(start_step):]: yield plan

    def fingerprint(self, epoch: int) -> str:
        """Stable digest of an epoch's plans, used to prove deterministic replay."""
        payload = [[list(plan.real_live), list(plan.real_spoof), list(plan.synthetic)]
                   for plan in self.epoch_plans(epoch)]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def composition_of(kinds: Sequence[str]) -> dict[str, int]:
    out = {"real_live": 0, "real_spoof": 0, "synthetic_spoof": 0}
    for kind in kinds: out[kind] = out.get(kind, 0) + 1
    return out
