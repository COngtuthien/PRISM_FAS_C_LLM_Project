from __future__ import annotations
import hashlib, math
from dataclasses import dataclass
from typing import Iterator
import numpy as np
from .config import LoaderConfig, TRAINING_SPLIT
from .contracts import TargetIsolationViolation
from .package_index import PackageIndex

class SamplerConfigurationError(ValueError):
    """The sampler cannot honour the requested balance with this package."""
@dataclass(frozen=True)
class PoolStats:
    key:tuple[str,str]; size:int; draws:int; unique:int; reuse:int
class BalancedDomainClassBatchSampler:
    """Deterministic (dataset, label)-balanced batch sampler over source_train.

    Every batch takes an equal quota from each pool. Pools are shuffled
    independently per epoch and deterministically reshuffled when exhausted, so
    minority pools cycle without changing the per-batch composition.
    """
    def __init__(self,index:PackageIndex,config:LoaderConfig,*,epoch:int=0):
        if index.split!=TRAINING_SPLIT:
            raise TargetIsolationViolation(f"the balanced training sampler only accepts {TRAINING_SPLIT!r}, got {index.split!r}")
        policy=config.sampler
        self.index=index; self.config=config; self.policy=policy; self.epoch=epoch
        self.pools:dict[tuple[str,str],list[int]]={}
        for position,row in enumerate(index.rows):
            self.pools.setdefault((row["dataset"],row["label_live_spoof"]),[]).append(position)
        self.pool_keys=tuple(sorted(self.pools))
        if not self.pool_keys: raise SamplerConfigurationError("source_train produced no pools")
        if policy.batch_size%len(self.pool_keys)!=0:
            raise SamplerConfigurationError(f"batch_size {policy.batch_size} is not divisible by {len(self.pool_keys)} pools")
        self.per_pool=policy.batch_size//len(self.pool_keys)
        self.steps_per_epoch=policy.steps_per_epoch or math.ceil(len(index.rows)/policy.batch_size)
        self.records=tuple(row["source_record_id"] for row in index.rows)
        self._stats:dict[tuple[str,str],dict]={}
    def set_epoch(self,epoch:int)->None: self.epoch=int(epoch)
    def _generator(self,key:tuple[str,str])->np.random.Generator:
        seed_material=f"{self.index.content_identity}|{self.policy.seed}|{self.epoch}|{key[0]}|{key[1]}".encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(seed_material).digest()[:8],"big"))
    def _pool_order(self,key:tuple[str,str],needed:int)->list[int]:
        """Deterministic shuffled order, reshuffled and cycled until `needed` items."""
        generator=self._generator(key); members=self.pools[key]; order:list[int]=[]; cycles=0
        while len(order)<needed:
            permuted=list(np.asarray(members)[generator.permutation(len(members))])
            order.extend(int(value) for value in permuted); cycles+=1
        self._stats.setdefault(key,{})["cycles"]=cycles
        return order[:needed]
    def __len__(self)->int: return self.steps_per_epoch
    def state(self)->dict:
        return {"loader_schema_version":self.config.loader_schema_version,"package_content_identity":self.index.content_identity,
                "seed":self.policy.seed,"epoch":self.epoch,"batch_size":self.policy.batch_size,
                "steps_per_epoch":self.steps_per_epoch,"pools":{f"{a}/{b}":len(v) for (a,b),v in sorted(self.pools.items())},
                "per_pool_quota":self.per_pool,"balance":self.policy.balance,
                "avoid_same_record_in_batch":self.policy.avoid_same_record_in_batch,
                "minority_policy":self.policy.minority_policy}
    def __iter__(self)->Iterator[list[int]]:
        needed=self.steps_per_epoch*self.per_pool
        streams={key:self._pool_order(key,needed*3 if self.policy.avoid_same_record_in_batch else needed) for key in self.pool_keys}
        cursors={key:0 for key in self.pool_keys}
        draws={key:0 for key in self.pool_keys}; unique={key:set() for key in self.pool_keys}
        for _ in range(self.steps_per_epoch):
            batch:list[int]=[]; used_records:set[str]=set(); used_ids:set[int]=set()
            for key in self.pool_keys:
                stream=streams[key]; taken=0; skipped:list[int]=[]
                while taken<self.per_pool:
                    if cursors[key]>=len(stream):
                        stream.extend(self._pool_order(key,needed)); # deterministic top-up
                    position=stream[cursors[key]]; cursors[key]+=1
                    record=self.records[position]
                    if position in used_ids or (self.policy.avoid_same_record_in_batch and record in used_records and len(skipped)<len(self.pools[key])):
                        skipped.append(position); continue
                    batch.append(position); used_ids.add(position); used_records.add(record)
                    draws[key]+=1; unique[key].add(position); taken+=1
                # deferred candidates stay available for later batches
                stream[cursors[key]:cursors[key]]=skipped
            if len(batch)!=self.policy.batch_size:
                raise SamplerConfigurationError(f"batch has {len(batch)} items, expected {self.policy.batch_size}")
            yield batch
        self._last_stats=[PoolStats(key,len(self.pools[key]),draws[key],len(unique[key]),draws[key]-len(unique[key])) for key in self.pool_keys]
    @property
    def last_epoch_stats(self)->list[PoolStats]: return getattr(self,"_last_stats",[])
def batch_fingerprint(batches:list[list[str]])->str:
    """Stable hash of an epoch's batches, used to prove deterministic replay."""
    payload="|".join(",".join(batch) for batch in batches).encode()
    return hashlib.sha256(payload).hexdigest()
