"""M10 reliability / shortcut / causal tests (spec 17.3), executed on SOURCE data.

None of these tests opens a target label, and none of them opens the target FEATURE
package either: they ask whether the detector reads the physics it claims to read,
which is a question about the source domain and the frozen synthetic bank.

Every acceptance rule in `ACCEPTANCE` is DECLARED IN THIS FILE, before any test ran,
and is recorded beside each result. The declared tests already fixed what each test
measures and what a failure would mean (`prism_fas.evaluation.reliability`); what
the frozen declarations left as prose — "below the declared ceiling", "no systematic
large increase" — is given a number here, once, in advance. A test that runs and
fails stays FAILED and is reported as a negative result; it is never re-tuned until
it passes.

The frozen B08 reference checkpoint is the subject of every test that needs a
model. Nothing here trains, and no checkpoint or calibration is written.
"""
from __future__ import annotations
import json
from pathlib import Path

import modal

APP_NAME = "prism-fas-b-m10-reliability"
DATA_VOLUME, MODELS_VOLUME, RUNS_VOLUME = "prism-fas-b-data", "prism-fas-b-models", "prism-fas-b-runs"
DATA_MOUNT, MODELS_MOUNT, RUNS_MOUNT = "/vol/data", "/vol/models", "/vol/runs"

REMOTE_PACKAGE = f"{DATA_MOUNT}/packages/prism_data_v1_m3b"
REMOTE_BANK = f"{DATA_MOUNT}/synthetic_banks/prism_synthetic_bank_m8_v3_e84c78cd2a9b"
REMOTE_WEIGHT_ROOT = f"{MODELS_MOUNT}/pretrained/m9"
REMOTE_RUNS_ROOT = f"{RUNS_MOUNT}/runs"
REMOTE_CACHE_ROOT = f"{RUNS_MOUNT}/m9_cache"

REFERENCE_CHECKPOINT = "m9_reference_seed20260806/checkpoints/best.pt"
REFERENCE_CALIBRATION = "m9_reference_seed20260806/calibration/source_dev.json"
GPAT_ONLY_CHECKPOINT = "A03-synthetic_route-gpat_only-s20260806/checkpoints/best.pt"
PHYSICS_ONLY_CHECKPOINT = "A03-synthetic_route-physics_only-s20260806/checkpoints/best.pt"

EXPECTED_ARCHITECTURE_IDENTITY = "d9507e42abf8c1930f835f50635ce2a7b74d90504d659ba6cc9356ea83f26aa0"

# --- ACCEPTANCE RULES, DECLARED BEFORE ANY TEST RAN ---------------------------
# Each rule turns the frozen test's prose pass-rule into a number. They are fixed
# here, recorded in every result, and not revisited after seeing an outcome.
ACCEPTANCE = {
    "synthetic_vs_real_spoof_probe": {
        "metric": "held-out balanced accuracy of a linear probe on the detector's own evidence "
                  "vector (p_global, s_region, nine normalized regional distances)",
        "ceiling": 0.75,
        "rule": "PASS when balanced accuracy <= 0.75; a probe that separates synthetic from real "
                "spoof much better than chance means a generator fingerprint the detector could "
                "shortcut on"},
    "benign_jpeg_corruption": {
        "metric": "paired shift of decision_score on unchanged bona-fide content",
        "mean_shift_ceiling": 0.05, "p95_shift_ceiling": 0.15,
        "rule": "PASS when mean shift <= 0.05 AND p95 shift <= 0.15"},
    "benign_resize_corruption": {
        "metric": "paired shift of decision_score on unchanged bona-fide content",
        "mean_shift_ceiling": 0.05, "p95_shift_ceiling": 0.15,
        "rule": "PASS when mean shift <= 0.05 AND p95 shift <= 0.15"},
    "benign_color_corruption": {
        "metric": "paired shift of decision_score on unchanged bona-fide content",
        "mean_shift_ceiling": 0.05, "p95_shift_ceiling": 0.15,
        "rule": "PASS when mean shift <= 0.05 AND p95 shift <= 0.15"},
    "residual_scale_zero": {
        "metric": "decision_score (calibrated p_global) and mean regional distance as the GPAT/physics residual is scaled "
                  "from 1.0 down to 0.0 (0.0 is exactly the unedited live image)",
        "minimum_score_drop": 0.10,
        "rule": "PASS when decision_score falls by at least 0.10 from scale 1.0 to scale 0.0 AND the mean "
                "regional distance falls; a score that does not move means the decision does not "
                "depend on the artefact it claims to detect"},
    "recipe_region_shift": {
        "metric": "for two accepted samples sharing a live target and route but attacking DIFFERENT "
                  "regions, whether the peak regional distance moves with the attacked region",
        "minimum_agreement": 0.50,
        "rule": "PASS when the peak regional evidence lies in an attacked region for at least 50% of "
                "pairs and moves between the two members more often than it stays"},
    "artifact_map_swap": {
        "metric": "agreement between the local patch logits and the artefact map, correct vs swapped",
        "minimum_drop": 0.02,
        "rule": "PASS when agreement drops by at least 0.02 under a mismatched artefact map; no drop "
                "means the local head ignores the supervision it is given"},
    "cross_route_synthetic": {
        "metric": "mean decision_score on each route accepted samples, for the two frozen single-route "
                  "checkpoints (A03 gpat_only and A03 physics_only)",
        "minimum_retention": 0.70,
        "rule": "PASS when each single-route model retains at least 70% of its own-route mean spoof "
                "score on the other route"},
}

IMAGE_SIZE = 224

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy==2.1.3", "pyarrow==18.1.0",
        "opencv-python-headless==4.10.0.84", "timm==1.0.11", "transformers==4.49.0",
        "safetensors==0.4.5", "sentencepiece==0.2.0", "pydantic==2.10.3", "PyYAML==6.0.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": "/root/project/src",
          "PYTHONIOENCODING": "utf-8"})
    .add_local_dir("src", "/root/project/src")
    .add_local_dir("configs", "/root/project/configs")
    .add_local_dir("assets", "/root/project/assets")
)
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME)
models_volume = modal.Volume.from_name(MODELS_VOLUME)
runs_volume = modal.Volume.from_name(RUNS_VOLUME)
VOLUMES = {DATA_MOUNT: data_volume, MODELS_MOUNT: models_volume, RUNS_MOUNT: runs_volume}


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=6 * 3600, memory=32768)
def m10_reliability(tests: list[str] | None = None, limit: int = 320) -> dict:
    """Run the declared reliability tests on SOURCE data with frozen checkpoints."""
    import numpy as np
    import torch
    from prism_fas.data.loader.config import load_loader_config
    from prism_fas.data.loader.loose_dataset import CanonicalPackageDataset
    from prism_fas.detector.config import detector_config_from, load_yaml
    from prism_fas.detector.contracts import DetectorBatch, REGION_ORDER
    from prism_fas.detector.heads import resolve_recipe_text_cache
    from prism_fas.detector.pretrained import SigLIP2Artifacts, resolve_convnext_weight
    from prism_fas.detector.prism_detector import build_detector
    from prism_fas.detector.region_cache import load_or_build_region_prior_cache
    from prism_fas.detector.synthetic_bank import SyntheticBankReader
    from prism_fas.evaluation import target_prediction as g7

    project = Path("/root/project")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wanted = set(tests or list(ACCEPTANCE))

    def make_model(checkpoint_relative: str, variant=None):
        from prism_fas.detector.variant import ResolvedExperimentVariant
        resolved = variant or ResolvedExperimentVariant.reference()
        payload = load_yaml(project / "configs/models/m9_detector.yaml")
        config = detector_config_from(payload, resolved)
        text_cache = resolve_recipe_text_cache(Path(REMOTE_WEIGHT_ROOT))
        model = build_detector(config, text_embeddings=text_cache.tensor(),
                               siglip=SigLIP2Artifacts.resolve(Path(REMOTE_WEIGHT_ROOT)),
                               local_weight_file=resolve_convnext_weight(Path(REMOTE_WEIGHT_ROOT)),
                               text_cache_identity=text_cache.identity, device=device)
        opened = g7.load_checkpoint_for_inference(Path(REMOTE_RUNS_ROOT) / checkpoint_relative, model)
        return model, opened

    model, opened = make_model(REFERENCE_CHECKPOINT)
    calibration = json.loads((Path(REMOTE_RUNS_ROOT) / REFERENCE_CALIBRATION).read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    threshold = float(calibration["selected_threshold"])

    loader_config = load_loader_config(project / "configs/data/loader_m4.yaml")

    # Each loader mode permits exactly one split, which is how target isolation is
    # enforced at the loader: `inference` is the only mode that can reach
    # `target_test`, and nothing here ever asks for it.
    MODE_FOR_SPLIT = {"source_train": "training", "source_dev": "validation"}

    def split_rows(split: str, *, label: str | None = None, count: int | None = None):
        dataset = CanonicalPackageDataset(Path(REMOTE_PACKAGE), split, loader_config,
                                          mode=MODE_FOR_SPLIT[split])
        rows = [row for row in dataset.index.rows
                if label is None or row.get("label_live_spoof") == label]
        cache, _ = load_or_build_region_prior_cache(
            Path(REMOTE_CACHE_ROOT), Path(REMOTE_PACKAGE), dataset.index.rows,
            package_identity=dataset.index.content_identity, split=split)
        positions = {row["sample_id"]: index for index, row in enumerate(dataset.index.rows)}
        selected = rows[:count] if count else rows
        return dataset, cache, positions, selected

    def forward(images: np.ndarray, priors: np.ndarray, visibility: np.ndarray, ids: tuple[str, ...]):
        """One label-free forward pass. The label field is a structural requirement of
        `DetectorBatch` and is never read by inference; it is set to live and the
        model output is what is measured."""
        batch = DetectorBatch(
            image=torch.from_numpy(images), label=torch.zeros(len(ids), dtype=torch.long),
            dataset_id=torch.zeros(len(ids), dtype=torch.long),
            is_synthetic=torch.zeros(len(ids), dtype=torch.bool),
            region_priors=torch.from_numpy(priors), visibility=torch.from_numpy(visibility),
            sample_ids=ids, datasets=("casia_fasd",)).validate().to(device)
        with torch.inference_mode():
            output = model(batch)
        logit = output.global_logit.detach().float().cpu().numpy().reshape(-1)
        p_global = 1.0 / (1.0 + np.exp(-logit / temperature))
        s_region = (output.s_region.detach().float().cpu().numpy()
                    if output.s_region is not None else np.zeros_like(p_global))
        distances = (output.aux["normalized_distances"].detach().float().cpu().numpy()
                     if "normalized_distances" in output.aux else np.zeros((len(ids), 9)))
        local = (output.local_logits.detach().float().cpu().numpy()
                 if output.local_logits is not None else None)
        s_final = 1.0 - (1.0 - p_global) * (1.0 - s_region)
        # `decision_score` is the calibrated p_global — the quantity the frozen G6
        # threshold belongs to (target evaluation contract section 2b). Every pass
        # rule below is stated on it, so a reliability result and a reported metric
        # are about the same number.
        return {"p_global": p_global, "s_region": s_region, "s_final": s_final,
                "decision_score": p_global, "distances": distances, "local_logits": local}

    def run_batched(images, priors, visibility, ids, size: int = 16):
        blocks = []
        for start in range(0, len(ids), size):
            window = slice(start, start + size)
            blocks.append(forward(images[window], priors[window], visibility[window], ids[window]))
        return {key: (np.concatenate([block[key] for block in blocks])
                      if blocks and blocks[0][key] is not None else None)
                for key in blocks[0]} if blocks else {}

    def source_arrays(split: str, label: str | None, count: int):
        dataset, cache, positions, selected = split_rows(split, label=label, count=count)
        images, priors, visible, ids = [], [], [], []
        for row in selected:
            position = positions[row["sample_id"]]
            sample = dataset[position]
            images.append(np.asarray(sample.image, dtype=np.float32))
            priors.append(cache.prior(position)); visible.append(cache.visible(position))
            ids.append(sample.sample_id)
        return (np.stack(images), np.stack(priors), np.stack(visible), tuple(ids),
                dataset, cache, positions, selected)

    results: dict = {}

    # --- benign corruptions, source_dev LIVE only ----------------------------
    def jpeg(images: np.ndarray) -> np.ndarray:
        import cv2
        out = np.empty_like(images)
        for index, item in enumerate(images):
            bgr = (np.transpose(item, (1, 2, 0))[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)
            ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            out[index] = np.transpose(decoded[:, :, ::-1].astype(np.float32) / 255.0, (2, 0, 1))
        return out

    def resize(images: np.ndarray) -> np.ndarray:
        import cv2
        out = np.empty_like(images)
        for index, item in enumerate(images):
            rgb = np.transpose(item, (1, 2, 0))
            small = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
            back = cv2.resize(small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            out[index] = np.transpose(back, (2, 0, 1))
        return out

    def colour(images: np.ndarray) -> np.ndarray:
        # A benign camera/exposure difference: +6% brightness, 0.94x contrast about
        # the mid grey. Well inside what an unchanged bona-fide capture can vary by.
        shifted = np.clip((images - 0.5) * 0.94 + 0.5 + 0.06, 0.0, 1.0)
        return shifted.astype(np.float32)

    corruptions = {"benign_jpeg_corruption": jpeg, "benign_resize_corruption": resize,
                   "benign_color_corruption": colour}
    if wanted & set(corruptions):
        images, priors, visible, ids, *_ = source_arrays("source_dev", "live", limit)
        clean = run_batched(images, priors, visible, ids)
        for name, transform in corruptions.items():
            if name not in wanted: continue
            after = run_batched(transform(images), priors, visible, ids)
            delta = after["decision_score"] - clean["decision_score"]
            rule = ACCEPTANCE[name]
            measurement = {"samples": int(delta.size), "score": "decision_score (calibrated p_global)",
                           "mean_before": float(clean["decision_score"].mean()),
                           "mean_after": float(after["decision_score"].mean()),
                           "mean_fused_before": float(clean["s_final"].mean()),
                           "mean_fused_after": float(after["s_final"].mean()),
                           "mean_shift": float(delta.mean()),
                           "median_shift": float(np.median(delta)),
                           "p95_shift": float(np.percentile(delta, 95)),
                           "max_shift": float(delta.max()),
                           "fraction_increased": float((delta > 0).mean()),
                           "population": "source_dev LIVE only; no spoof, no target"}
            results[name] = {"acceptance": rule, "result": measurement,
                             "passed": bool(measurement["mean_shift"] <= rule["mean_shift_ceiling"]
                                            and measurement["p95_shift"] <= rule["p95_shift_ceiling"])}

    # --- the synthetic bank -------------------------------------------------
    needs_bank = wanted & {"synthetic_vs_real_spoof_probe", "residual_scale_zero",
                           "recipe_region_shift", "artifact_map_swap", "cross_route_synthetic"}
    if needs_bank:
        bank = SyntheticBankReader.open(Path(REMOTE_BANK))
        dataset, cache, positions, _ = split_rows("source_train")
        by_sample = {row["sample_id"]: index for index, row in enumerate(dataset.index.rows)}

        def synthetic_arrays(indices):
            images, priors, visible, ids, meta = [], [], [], [], []
            for index in indices:
                sample = bank.sample(index)
                live_position = by_sample.get(sample.live_target_sample_id)
                if live_position is None: continue
                images.append(np.asarray(sample.image, dtype=np.float32))
                priors.append(cache.prior(live_position)); visible.append(cache.visible(live_position))
                ids.append(sample.synthetic_id)
                meta.append({"sample": sample, "live_position": live_position})
            return np.stack(images), np.stack(priors), np.stack(visible), tuple(ids), meta

        accepted = list(range(len(bank)))[:limit]

        if "synthetic_vs_real_spoof_probe" in wanted:
            s_images, s_priors, s_visible, s_ids, _ = synthetic_arrays(accepted)
            synthetic_out = run_batched(s_images, s_priors, s_visible, s_ids)
            r_images, r_priors, r_visible, r_ids, *_ = source_arrays("source_train", "spoof", limit)
            real_out = run_batched(r_images, r_priors, r_visible, r_ids)
            features = np.concatenate([
                np.column_stack([synthetic_out["p_global"], synthetic_out["s_region"],
                                 synthetic_out["distances"]]),
                np.column_stack([real_out["p_global"], real_out["s_region"], real_out["distances"]])])
            targets = np.concatenate([np.ones(len(s_ids)), np.zeros(len(r_ids))])
            # A deterministic, seeded half/half split and a plain logistic regression
            # by gradient descent — no sklearn in the image, and a probe this simple
            # cannot be accused of finding the separation by capacity alone.
            generator = np.random.Generator(np.random.PCG64(20260810))
            order = generator.permutation(len(targets))
            split = len(order) // 2
            train, test = order[:split], order[split:]
            mean, std = features[train].mean(0), features[train].std(0) + 1e-8
            x_train = (features[train] - mean) / std
            x_test = (features[test] - mean) / std
            weights = np.zeros(x_train.shape[1] + 1)
            design = np.column_stack([x_train, np.ones(len(x_train))])
            for _ in range(4000):
                prediction = 1.0 / (1.0 + np.exp(-design @ weights))
                weights -= 0.5 * design.T @ (prediction - targets[train]) / len(design)
            scores = 1.0 / (1.0 + np.exp(-np.column_stack([x_test, np.ones(len(x_test))]) @ weights))
            predicted = (scores >= 0.5).astype(np.float64)
            truth = targets[test]
            sensitivity = float((predicted[truth == 1] == 1).mean()) if (truth == 1).any() else 0.0
            specificity = float((predicted[truth == 0] == 0).mean()) if (truth == 0).any() else 0.0
            balanced = (sensitivity + specificity) / 2.0
            rule = ACCEPTANCE["synthetic_vs_real_spoof_probe"]
            results["synthetic_vs_real_spoof_probe"] = {
                "acceptance": rule,
                "result": {"balanced_accuracy": balanced, "sensitivity_synthetic": sensitivity,
                           "specificity_real_spoof": specificity,
                           "synthetic_samples": len(s_ids), "real_spoof_samples": len(r_ids),
                           "held_out_samples": int(len(test)),
                           "population": "source_train real spoof vs accepted M8 v3 synthetic; no target"},
                "passed": bool(balanced <= rule["ceiling"])}

        if "residual_scale_zero" in wanted:
            s_images, s_priors, s_visible, s_ids, meta = synthetic_arrays(accepted)
            live = np.stack([np.asarray(dataset[item["live_position"]].image, dtype=np.float32)
                             for item in meta])
            sweep = {}
            for scale in (0.0, 0.25, 0.5, 1.0):
                blended = np.clip(live + float(scale) * (s_images - live), 0.0, 1.0).astype(np.float32)
                block = run_batched(blended, s_priors, s_visible, s_ids)
                sweep[str(scale)] = {"mean_decision_score": float(block["decision_score"].mean()),
                                     "mean_s_final": float(block["s_final"].mean()),
                                     "mean_s_region": float(block["s_region"].mean()),
                                     "mean_regional_distance": float(block["distances"].mean())}
            drop = sweep["1.0"]["mean_decision_score"] - sweep["0.0"]["mean_decision_score"]
            distance_drop = (sweep["1.0"]["mean_regional_distance"]
                             - sweep["0.0"]["mean_regional_distance"])
            rule = ACCEPTANCE["residual_scale_zero"]
            results["residual_scale_zero"] = {
                "acceptance": rule,
                "result": {"sweep": sweep, "score_drop_from_1_to_0": float(drop),
                           "regional_distance_drop": float(distance_drop), "samples": len(s_ids),
                           "population": "accepted M8 v3 synthetic blended back to their own live "
                                         "targets; scale 0.0 IS the unedited live image; no target"},
                "passed": bool(drop >= rule["minimum_score_drop"] and distance_drop > 0.0)}

        if "recipe_region_shift" in wanted:
            grouped: dict[str, list[int]] = {}
            for index, row in enumerate(bank.rows):
                grouped.setdefault(str(row["live_target_sample_id"]), []).append(index)
            pairs = []
            for indices in grouped.values():
                for left in range(len(indices)):
                    for right in range(left + 1, len(indices)):
                        a, b = bank.rows[indices[left]], bank.rows[indices[right]]
                        if a["route"] == b["route"] and set(str(a["regions"]).split("|")) != \
                                set(str(b["regions"]).split("|")):
                            pairs.append((indices[left], indices[right]))
            pairs = pairs[:min(len(pairs), max(16, limit // 4))]
            if not pairs:
                results["recipe_region_shift"] = {
                    "acceptance": ACCEPTANCE["recipe_region_shift"], "status": "BLOCKED",
                    "blocked_reason": "the frozen bank holds no two accepted samples that share a "
                                      "live target and a route while attacking different regions"}
            else:
                flat = [index for pair in pairs for index in pair]
                images, priors, visible, ids, meta = synthetic_arrays(flat)
                block = run_batched(images, priors, visible, ids)
                peaks = block["distances"].argmax(axis=1)
                in_attacked, moved = 0, 0
                for position in range(0, len(ids) - 1, 2):
                    left_regions = set(str(meta[position]["sample"].regions).split("|")) \
                        if isinstance(meta[position]["sample"].regions, str) \
                        else set(meta[position]["sample"].regions)
                    right_regions = set(str(meta[position + 1]["sample"].regions).split("|")) \
                        if isinstance(meta[position + 1]["sample"].regions, str) \
                        else set(meta[position + 1]["sample"].regions)
                    left_peak = REGION_ORDER[int(peaks[position])]
                    right_peak = REGION_ORDER[int(peaks[position + 1])]
                    in_attacked += int(left_peak in left_regions) + int(right_peak in right_regions)
                    moved += int(left_peak != right_peak)
                total = len(ids)
                agreement = in_attacked / total if total else 0.0
                movement = moved / (total // 2) if total else 0.0
                rule = ACCEPTANCE["recipe_region_shift"]
                results["recipe_region_shift"] = {
                    "acceptance": rule,
                    "result": {"pairs": total // 2, "peak_in_attacked_region_rate": float(agreement),
                               "peak_moved_between_members_rate": float(movement),
                               "population": "accepted M8 v3 samples sharing a live target and route "
                                             "but attacking different regions; no target"},
                    "passed": bool(agreement >= rule["minimum_agreement"] and movement > 0.5)}

        if "artifact_map_swap" in wanted:
            s_images, s_priors, s_visible, s_ids, meta = synthetic_arrays(accepted)
            block = run_batched(s_images, s_priors, s_visible, s_ids)
            local = block.get("local_logits")
            if local is None:
                results["artifact_map_swap"] = {
                    "acceptance": ACCEPTANCE["artifact_map_swap"], "status": "BLOCKED",
                    "blocked_reason": "the reference detector produced no local patch logits, so "
                                      "there is no local supervision to mismatch"}
            else:
                side = int(round(local.shape[1] ** 0.5))
                maps = np.stack([item["sample"].artifact_map[0] for item in meta])
                pooled = maps.reshape(len(maps), side, IMAGE_SIZE // side, side,
                                      IMAGE_SIZE // side).mean(axis=(2, 4)).reshape(len(maps), -1)
                probability = 1.0 / (1.0 + np.exp(-local))

                def agreement(target: np.ndarray) -> float:
                    """Mean correlation between the local response and the artefact map."""
                    values = []
                    for prediction, truth in zip(probability, target):
                        if truth.std() < 1e-8 or prediction.std() < 1e-8: continue
                        values.append(float(np.corrcoef(prediction, truth)[0, 1]))
                    return float(np.mean(values)) if values else 0.0

                # The swap is a DERANGEMENT: every sample gets another sample's map.
                swapped = np.roll(pooled, 1, axis=0)
                correct, mismatched = agreement(pooled), agreement(swapped)
                rule = ACCEPTANCE["artifact_map_swap"]
                results["artifact_map_swap"] = {
                    "acceptance": rule,
                    "result": {"agreement_correct_map": correct,
                               "agreement_swapped_map": mismatched,
                               "drop": float(correct - mismatched), "samples": len(s_ids),
                               "patch_grid": side,
                               "population": "accepted M8 v3 synthetic samples with artefact maps "
                                             "cyclically swapped; no target"},
                    "passed": bool(correct - mismatched >= rule["minimum_drop"])}

        if "cross_route_synthetic" in wanted:
            from prism_fas.detector.variant import variant_from_row
            from prism_fas.evaluation.experiment_matrix import build_plan
            plan = build_plan(project / "configs/experiments/m10_matrix.yaml")
            rows = {row["experiment_id"]: row for row in plan["rows"]}
            by_route = {route: [index for index, row in enumerate(bank.rows) if row["route"] == route][:limit]
                        for route in ("gpat", "physics")}
            arrays = {route: synthetic_arrays(indices) for route, indices in by_route.items()}
            measured = {}
            for name, relative in (("gpat_only", GPAT_ONLY_CHECKPOINT),
                                   ("physics_only", PHYSICS_ONLY_CHECKPOINT)):
                row = rows[f"A03-synthetic_route-{name}-s20260806"]
                model, _ = make_model(relative, variant=variant_from_row(row))
                measured[name] = {route: float(run_batched(*arrays[route][:4])["decision_score"].mean())
                                  for route in ("gpat", "physics")}
            model, _ = make_model(REFERENCE_CHECKPOINT)      # restore the reference
            retention = {
                "gpat_only": (measured["gpat_only"]["physics"] / measured["gpat_only"]["gpat"]
                              if measured["gpat_only"]["gpat"] else 0.0),
                "physics_only": (measured["physics_only"]["gpat"] / measured["physics_only"]["physics"]
                                 if measured["physics_only"]["physics"] else 0.0)}
            rule = ACCEPTANCE["cross_route_synthetic"]
            results["cross_route_synthetic"] = {
                "acceptance": rule,
                "result": {"mean_s_final_by_model_and_route": measured,
                           "cross_route_retention": retention,
                           "samples_per_route": {route: len(indices) for route, indices in by_route.items()},
                           "population": "the two frozen A03 single-route checkpoints evaluated on both "
                                         "routes of the frozen bank; NOTHING was retrained; no target"},
                "passed": bool(min(retention.values()) >= rule["minimum_retention"])}

    return {"schema_version": "m10-reliability-execution-v1",
            "device": device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "reference_checkpoint_sha256": opened["checkpoint_sha256"],
            "architecture_identity": opened["identity"].get("architecture_identity"),
            "architecture_matches_pin":
                opened["identity"].get("architecture_identity") == EXPECTED_ARCHITECTURE_IDENTITY,
            "calibration_temperature": temperature, "calibration_threshold": threshold,
            "limit": int(limit), "tests": results,
            "acceptance_declared_before_running": True,
            "target_features_opened": False, "target_labels_opened": False,
            "trained_anything": False, "wrote_checkpoint": False}


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=2 * 3600, memory=32768)
def m10_decision_score_diagnostic(limit: int | None = None) -> dict:
    """Which quantity does the frozen source-dev threshold actually belong to?

    SOURCE-ONLY, and decisive. `M9Trainer.run_g6` calibrates temperature and selects
    the min-ACER threshold on `output.global_logit` alone, and `validate()` selects
    the checkpoint on `sigmoid(global_logit)`. So the frozen operating point is
    defined on the CALIBRATED p_global. The M10 target contract, however, declares
    the decision on the fused `s_final = 1 - (1 - p_global)(1 - s_region)`, which is
    pointwise >= p_global.

    This function measures both on the whole of `source_dev`, where labels are
    source-side and always legitimately available, so the question is settled from
    source evidence and never from a target observation.
    """
    import numpy as np
    import torch
    from prism_fas.data.loader.config import load_loader_config
    from prism_fas.data.loader.loose_dataset import CanonicalPackageDataset
    from prism_fas.detector.config import detector_config_from, load_yaml
    from prism_fas.detector.contracts import DetectorBatch
    from prism_fas.detector.heads import resolve_recipe_text_cache
    from prism_fas.detector.pretrained import SigLIP2Artifacts, resolve_convnext_weight
    from prism_fas.detector.prism_detector import build_detector
    from prism_fas.detector.region_cache import load_or_build_region_prior_cache
    from prism_fas.detector.variant import ResolvedExperimentVariant
    from prism_fas.evaluation import target_prediction as g7
    from prism_fas.train.metrics import apcer_bpcer_acer, equal_error_rate, roc_auc

    project = Path("/root/project")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = ResolvedExperimentVariant.reference()
    config = detector_config_from(load_yaml(project / "configs/models/m9_detector.yaml"), resolved)
    text_cache = resolve_recipe_text_cache(Path(REMOTE_WEIGHT_ROOT))
    model = build_detector(config, text_embeddings=text_cache.tensor(),
                           siglip=SigLIP2Artifacts.resolve(Path(REMOTE_WEIGHT_ROOT)),
                           local_weight_file=resolve_convnext_weight(Path(REMOTE_WEIGHT_ROOT)),
                           text_cache_identity=text_cache.identity, device=device)
    opened = g7.load_checkpoint_for_inference(Path(REMOTE_RUNS_ROOT) / REFERENCE_CHECKPOINT, model)
    calibration = json.loads((Path(REMOTE_RUNS_ROOT) / REFERENCE_CALIBRATION).read_text(encoding="utf-8"))
    temperature = float(calibration["temperature"])
    threshold = float(calibration["selected_threshold"])

    loader_config = load_loader_config(project / "configs/data/loader_m4.yaml")
    dataset = CanonicalPackageDataset(Path(REMOTE_PACKAGE), "source_dev", loader_config,
                                      mode="validation")
    rows = dataset.index.rows if limit is None else dataset.index.rows[:limit]
    cache, _ = load_or_build_region_prior_cache(
        Path(REMOTE_CACHE_ROOT), Path(REMOTE_PACKAGE), dataset.index.rows,
        package_identity=dataset.index.content_identity, split="source_dev")
    p_global, s_region, labels = [], [], []
    for start in range(0, len(rows), 16):
        window = list(range(start, min(start + 16, len(rows))))
        samples = [dataset[index] for index in window]
        batch = DetectorBatch(
            image=torch.from_numpy(np.stack([np.asarray(item.image, dtype=np.float32)
                                             for item in samples])),
            label=torch.tensor([int(item.class_target) for item in samples], dtype=torch.long),
            dataset_id=torch.zeros(len(window), dtype=torch.long),
            is_synthetic=torch.zeros(len(window), dtype=torch.bool),
            region_priors=torch.from_numpy(np.stack([cache.prior(index) for index in window])),
            visibility=torch.from_numpy(np.stack([cache.visible(index) for index in window])),
            sample_ids=tuple(item.sample_id for item in samples),
            datasets=("casia_fasd",)).validate().to(device)
        with torch.inference_mode():
            output = model(batch)
        logit = output.global_logit.detach().float().cpu().numpy().reshape(-1)
        p_global.append(1.0 / (1.0 + np.exp(-logit / temperature)))
        s_region.append(output.s_region.detach().float().cpu().numpy()
                        if output.s_region is not None else np.zeros(len(window)))
        labels.append(batch.label.detach().cpu().numpy())
    p_global = np.concatenate(p_global); s_region = np.concatenate(s_region)
    labels = np.concatenate(labels)
    s_final = 1.0 - (1.0 - p_global) * (1.0 - s_region)

    def block(scores: np.ndarray) -> dict:
        at = apcer_bpcer_acer(scores, labels, threshold)
        eer = equal_error_rate(scores, labels)
        return {"apcer": float(at["apcer"]), "bpcer": float(at["bpcer"]), "acer": float(at["acer"]),
                "roc_auc": float(roc_auc(scores, labels)), "eer": float(eer["eer"]),
                "eer_threshold": float(eer["threshold"]),
                "mean_live": float(scores[labels == 0].mean()),
                "mean_spoof": float(scores[labels == 1].mean())}

    return {"schema_version": "m10-decision-score-diagnostic-v1",
            "population": {"samples": int(labels.size), "live": int((labels == 0).sum()),
                           "spoof": int((labels == 1).sum()), "split": "source_dev"},
            "frozen_threshold": threshold, "frozen_temperature": temperature,
            "checkpoint_sha256": opened["checkpoint_sha256"],
            "calibrated_p_global": block(p_global),
            "fused_s_final": block(s_final),
            "mean_s_region_live": float(s_region[labels == 0].mean()),
            "mean_s_region_spoof": float(s_region[labels == 1].mean()),
            "frozen_g6_record_at_threshold": calibration.get("source_dev_metrics_at_threshold"),
            "recorded_best_source_dev": "see SOURCE_MATRIX_LOCK B08-s20260806",
            "target_features_opened": False, "target_labels_opened": False}


@app.local_entrypoint()
def main(tests: str = "", limit: int = 320, output: str = "reports/m10/RELIABILITY_EXECUTION.json",
         action: str = "tests") -> None:
    if action == "decision-score":
        payload = m10_decision_score_diagnostic.remote(None)
        Path("reports/m10/DECISION_SCORE_DIAGNOSTIC.json").write_text(
            json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(json.dumps(payload, indent=1, default=str))
        return
    selected = [name.strip() for name in tests.split(",") if name.strip()] or None
    payload = m10_reliability.remote(selected, limit)
    Path(output).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"written": output, "tests": sorted(payload["tests"]),
                      "passed": {name: block.get("passed", block.get("status"))
                                 for name, block in payload["tests"].items()}}, indent=1))
