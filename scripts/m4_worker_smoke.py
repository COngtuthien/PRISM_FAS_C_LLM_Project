"""Real DataLoader worker smoke for the M4 loader/sampler (Windows-safe)."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

PACKAGE = Path("data/processed/prism_data_v1_m3b")
CONFIG_PATH = Path("configs/data/loader_m4.yaml")


def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    from prism_fas.data.loader import (BalancedDomainClassBatchSampler, CanonicalPackageDataset, collate_source_batch,
                                       collate_target_batch, load_loader_config, open_package)
    config = load_loader_config(CONFIG_PATH)
    results = {}
    for workers in (0, 2):
        attempt = {"num_workers": workers}
        try:
            index = open_package(PACKAGE, "source_train", config, mode="training")
            train = CanonicalPackageDataset(PACKAGE, "source_train", config, mode="training", index=index)
            sampler = BalancedDomainClassBatchSampler(index, config)
            started = time.time(); batches = 0; samples = 0
            loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_source_batch, num_workers=workers)
            for batch in loader:
                batches += 1; samples += int(batch["target"].shape[0])
                assert batch["image"].shape[1:] == (3, 224, 224)
                if batches >= 10: break
            del loader
            attempt["source_train"] = {"batches": batches, "samples": samples, "seconds": round(time.time() - started, 2)}

            dev = CanonicalPackageDataset(PACKAGE, "source_dev", config, mode="validation")
            loader = DataLoader(dev, batch_size=32, shuffle=False, collate_fn=collate_source_batch, num_workers=workers)
            seen = []; batches = 0
            for batch in loader:
                seen.extend(batch["sample_id"]); batches += 1
                if batches >= 10: break
            del loader
            attempt["source_dev"] = {"batches": batches, "samples": len(seen), "unique": len(set(seen)),
                                     "sequential": seen == sorted(seen)}

            target = CanonicalPackageDataset(PACKAGE, "target_test", config, mode="inference")
            loader = DataLoader(target, batch_size=32, shuffle=False, collate_fn=collate_target_batch, num_workers=workers)
            seen = []; batches = 0; identity = 0
            for batch in loader:
                seen.extend(batch["sample_id"]); batches += 1
                identity += int(batch["identity_available"].sum())
                assert "target" not in batch and "label" not in batch
                if batches >= 10: break
            del loader
            attempt["target_test"] = {"batches": batches, "samples": len(seen), "identity_available": identity}
            attempt["status"] = "passed"
        except Exception as exc:                       # genuine attempt; report the real error
            attempt["status"] = "failed"; attempt["error"] = f"{type(exc).__name__}: {exc}"
        results[f"num_workers={workers}"] = attempt
    Path("reports/m4").mkdir(parents=True, exist_ok=True)
    Path("reports/m4/worker_smoke.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results))
    return 0 if all(entry["status"] == "passed" for entry in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
