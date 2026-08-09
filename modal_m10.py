"""M10 Modal wrapper: remote target-feature verification, label-free G7 prediction
and bounded source-only matrix training.

The wrapper only orchestrates. Every function calls the same core modules the local
path uses; `src/prism_fas/evaluation/**` and `src/prism_fas/detector/**` must never
import modal (spec Table 41 / section 12.1).

MOUNT FIREWALL. Training and G7 get DIFFERENT mount sets, and neither can resolve a
label:

    train  ->  /vol/data/packages/prism_data_v1_m3b          (source package)
               /vol/data/synthetic_banks/...m8_v3...         (M8 bank)
               /vol/models/pretrained/m9                     (pinned weights)
               /vol/runs                                     (run artifacts)
               TARGET FEATURES: not mounted

    g7     ->  /vol/data/target_eval/prism_target_eval_v2    (target FEATURES only)
               /vol/models/pretrained/m9
               /vol/runs
               SOURCE PACKAGE: not needed;  LABELS: nowhere on any volume

The sealed evaluator-only label artifact is NEVER uploaded to any Modal volume. G8
runs locally against the local label tree, which is git-ignored.

Volumes are REUSED, never duplicated.
"""
from __future__ import annotations
import json
from pathlib import Path

import modal

APP_NAME = "prism-fas-b-m10"
DATA_VOLUME, MODELS_VOLUME, RUNS_VOLUME = "prism-fas-b-data", "prism-fas-b-models", "prism-fas-b-runs"
DATA_MOUNT, MODELS_MOUNT, RUNS_MOUNT = "/vol/data", "/vol/models", "/vol/runs"

REMOTE_PACKAGE = f"{DATA_MOUNT}/packages/prism_data_v1_m3b"
REMOTE_BANK = f"{DATA_MOUNT}/synthetic_banks/prism_synthetic_bank_m8_v3_e84c78cd2a9b"
# The A02 control artifact: source-only, same candidate budget, same routes, same
# frozen quality thresholds and the same frozen generator weights — random operator
# composition instead of the structured recipe. Built by `modal_m10_a02.py`, and it
# lives on the runs volume where it was assembled; the training mount already
# includes that volume, so it is read in place rather than copied.
REMOTE_RANDOM_BANK = (f"{RUNS_MOUNT}/synthetic_banks/m10_a02_work/generation/"
                      f"prism_synthetic_bank_m10_random_f7f1e6ac2034")
REMOTE_TARGET_FEATURES = f"{DATA_MOUNT}/target_eval/prism_target_eval_v2"
REMOTE_WEIGHT_ROOT = f"{MODELS_MOUNT}/pretrained/m9"
REMOTE_RUNS_ROOT = f"{RUNS_MOUNT}/runs"
REMOTE_CACHE_ROOT = f"{RUNS_MOUNT}/m9_cache"
REMOTE_M10_CACHE = f"{RUNS_MOUNT}/m10_cache"

EXPECTED_SOURCE_PACKAGE_IDENTITY = "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
EXPECTED_TARGET_FEATURE_IDENTITY = "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8"
EXPECTED_TARGET_PACKAGE_ID = "prism_target_eval_v2"
EXPECTED_TARGET_ROWS = 6776
EXPECTED_TARGET_VIDEOS = 1700
EXPECTED_BANK_IDENTITY = "e84c78cd2a9b548244e243de0380998d04bc6770b91caf32ac7be96f489bb542"
EXPECTED_SIGLIP2_IDENTITY = "7e059e40dcc34913b51fc8d7bd25e6f0c023bc238261effee9bfb87b33f04822"
EXPECTED_TEXT_CACHE_IDENTITY = "10f4ec35b7563b2b658cacc94599d35b9f93b531963a065459d4694d5dc2c141"
EXPECTED_ARCHITECTURE_IDENTITY = "d9507e42abf8c1930f835f50635ce2a7b74d90504d659ba6cc9356ea83f26aa0"
EXPECTED_MATRIX_IDENTITY = "a4972b0dc23946c4ad169f2c856fc9b5e0387baca45b2c9a4895f8180d9c2dd5"
# The frozen A02 control bank. `SyntheticBankReader.open` is fail-closed on both,
# so an A02 row can never silently fall back to the structured M8 v3 bank.
EXPECTED_RANDOM_BANK_ID = "prism_synthetic_bank_m10_random_f7f1e6ac2034"
EXPECTED_RANDOM_BANK_IDENTITY = "f7f1e6ac20341d32d75dddd19cbf3231ea4eb7554eb49290aee32cf59ec17387"

# A target feature row may never carry any of these, remotely or locally.
FORBIDDEN_FEATURE_FIELDS = (
    "label", "label_live_spoof", "true_label", "target", "class_target", "attack_type",
    "attack_family", "taxonomy", "private_label", "subject_id", "official_split",
    "identity_embedding", "source_path")
# `content_identity_sha256` is recomputed over the lock body with these excluded.
IDENTITY_EXCLUDED_FIELDS = ("created_at", "git_commit", "content_identity_sha256",
                            "build_seconds", "environment")

GPU_ALLOW_LIST = ("L4", "L40S")
DEFAULT_GPU = "L4"

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

# G7 and verification mount the data volume read-only and never touch /packages.
TARGET_VOLUMES = {DATA_MOUNT: data_volume, MODELS_MOUNT: models_volume, RUNS_MOUNT: runs_volume}
TRAIN_VOLUMES = {DATA_MOUNT: data_volume, MODELS_MOUNT: models_volume, RUNS_MOUNT: runs_volume}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""): digest.update(block)
    return digest.hexdigest()


# --- remote target feature verification ---------------------------------------

@app.function(image=image, volumes=TARGET_VOLUMES, timeout=3600, cpu=4.0, memory=16384)
def m10_verify_target_features() -> dict:
    """Re-derive the uploaded target FEATURE package inside the container.

    Fail-closed: every shard hash, every manifest hash and the content identity are
    recomputed from the bytes that actually landed on the volume and compared
    against the lock. A mismatch is an error, never a warning.
    """
    import hashlib
    import pyarrow.parquet as pq

    root = Path(REMOTE_TARGET_FEATURES)
    errors: list[str] = []
    lock = json.loads((root / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))

    # 1. shard bytes
    shard_results = []
    for entry in lock.get("shards") or []:
        path = root / "shards" / entry["shard_filename"]
        if not path.is_file():
            errors.append(f"missing shard {entry['shard_filename']}"); continue
        digest, size = _sha256(path), path.stat().st_size
        ok = digest == entry["sha256"] and size == entry["byte_size"]
        if not ok: errors.append(f"shard mismatch {entry['shard_filename']}")
        shard_results.append({"shard": entry["shard_filename"], "rows": entry["row_count"],
                              "sha256_matches": digest == entry["sha256"],
                              "byte_size_matches": size == entry["byte_size"]})

    # 2. manifest bytes
    manifest_results = {}
    for name, expected in (lock.get("manifest_sha256") or {}).items():
        path = root / "manifests" / f"{name}.parquet"
        if not path.is_file():
            errors.append(f"missing manifest {name}"); continue
        matches = _sha256(path) == expected
        manifest_results[name] = matches
        if not matches: errors.append(f"manifest hash mismatch {name}")

    # 3. the lock must hash to its own recorded content identity
    body = {key: value for key, value in lock.items() if key not in IDENTITY_EXCLUDED_FIELDS}
    recomputed = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                           default=str).encode("utf-8")).hexdigest()
    identity_self_consistent = recomputed == lock.get("content_identity_sha256")
    identity_matches_pin = lock.get("content_identity_sha256") == EXPECTED_TARGET_FEATURE_IDENTITY
    if not identity_self_consistent: errors.append("the remote lock does not hash to its own identity")
    if not identity_matches_pin: errors.append("the remote feature identity does not match the pin")
    if lock.get("package_id") != EXPECTED_TARGET_PACKAGE_ID: errors.append("unexpected package id")
    if lock.get("status") != "validated": errors.append(f"package status is {lock.get('status')!r}")

    # 4. the feature manifest itself: counts, privacy, priors
    features = pq.read_table(root / "manifests" / "target_test_features.parquet")
    columns = set(features.column_names)
    leaked = sorted(columns & set(FORBIDDEN_FEATURE_FIELDS))
    if leaked: errors.append(f"feature manifest exposes forbidden columns: {leaked}")
    rows = features.num_rows
    videos = len(set(features.column("source_record_id").to_pylist()))
    if rows != EXPECTED_TARGET_ROWS: errors.append(f"feature rows {rows} != {EXPECTED_TARGET_ROWS}")
    if videos != EXPECTED_TARGET_VIDEOS: errors.append(f"videos {videos} != {EXPECTED_TARGET_VIDEOS}")

    samples = pq.read_table(root / "manifests" / "samples.parquet").to_pylist()
    identity_computed = sum(1 for row in samples if row.get("identity_status") == "computed")
    if identity_computed: errors.append(f"{identity_computed} target samples carry an identity embedding")
    prior_counts = {name: sum(1 for row in samples if row.get(f"{name}_status") == "computed")
                    for name in ("parsing", "pose", "visibility")}
    for name, count in prior_counts.items():
        if count != EXPECTED_TARGET_ROWS: errors.append(f"{name} priors {count} != {EXPECTED_TARGET_ROWS}")

    # 5. the label artifact must not exist anywhere on any mounted volume
    label_hits = []
    for mount in (DATA_MOUNT, MODELS_MOUNT, RUNS_MOUNT):
        for pattern in ("**/siw_target_labels.parquet", "**/TARGET_LABEL_LOCK.json",
                        "**/evaluation_only/**"):
            label_hits += [str(path) for path in Path(mount).glob(pattern)]
    if label_hits: errors.append(f"a label artifact is resolvable remotely: {label_hits[:5]}")

    return {"passed": not errors, "errors": errors,
            "remote_root": str(root),
            "package_id": lock.get("package_id"), "status": lock.get("status"),
            "content_identity_sha256": lock.get("content_identity_sha256"),
            "identity_matches_pin": identity_matches_pin,
            "identity_self_consistent": identity_self_consistent,
            "shards": shard_results, "shard_count": len(shard_results),
            "shard_rows_total": sum(entry["rows"] for entry in shard_results),
            "manifests_verified": manifest_results,
            "feature_rows": rows, "videos": videos,
            "prior_counts": prior_counts, "target_identity_embeddings": identity_computed,
            "feature_label_leakage": len(leaked),
            "label_artifact_resolvable": bool(label_hits),
            "target_labels_opened": False}


# --- label-free G7 prediction ---------------------------------------------------

@app.function(image=image, gpu=DEFAULT_GPU, volumes=TARGET_VOLUMES, timeout=10800)
def m10_g7_predict(experiment_id: str, checkpoint_relative: str, variant: str = "B08",
                   seed: int | None = 20260806, threshold: float = 0.5,
                   temperature: float | None = None, calibration_hash: str = "",
                   calibration_sha256: str = "", limit: int | None = None,
                   batch_size: int = 32, engineering_smoke: bool = True,
                   cover_short_videos: int = 0, cover_full_videos: int = 0) -> dict:
    """G7 on the remote target FEATURE package. No label is reachable from here.

    The source package is not mounted for this stage's purposes and is never opened;
    the only data root this function touches is the target feature package.
    """
    import torch
    from prism_fas.data.loader.config import load_loader_config
    from prism_fas.detector.config import detector_config_from, load_yaml
    from prism_fas.detector.heads import resolve_recipe_text_cache
    from prism_fas.detector.pretrained import SigLIP2Artifacts, resolve_convnext_weight
    from prism_fas.detector.prism_detector import architecture_delta, build_detector
    from prism_fas.evaluation import target_prediction as g7
    from prism_fas.evaluation.firewall import TargetLabelFirewall, load_firewall_config

    project = Path("/root/project")
    firewall = TargetLabelFirewall(load_firewall_config(project / "configs/evaluation/m10_target.yaml"),
                                   project_root=project)
    evidence = firewall.assert_cannot_resolve_labels("G7")

    # The variant the checkpoint was trained under, re-derived from the frozen
    # matrix. G7 must understand the simpler baselines: a row without a regional
    # branch or a PromptHead writes `null` for the components it does not have, and
    # B00-B07 are never forced through the B08 fusion.
    from prism_fas.detector.variant import ResolvedExperimentVariant
    try:
        resolved = resolve_matrix_row(experiment_id)["variant"]
    except RuntimeError:
        # An engineering smoke uses an id that is not a matrix row; it is the
        # reference configuration by construction and is labelled as a smoke below.
        if not engineering_smoke: raise
        resolved = ResolvedExperimentVariant.reference()

    model_payload = load_yaml(project / "configs/models/m9_detector.yaml")
    detector_config = detector_config_from(model_payload, resolved)
    siglip = SigLIP2Artifacts.resolve(Path(REMOTE_WEIGHT_ROOT))
    text_cache = resolve_recipe_text_cache(Path(REMOTE_WEIGHT_ROOT))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_detector(detector_config, text_embeddings=text_cache.tensor(), siglip=siglip,
                           local_weight_file=resolve_convnext_weight(Path(REMOTE_WEIGHT_ROOT)),
                           text_cache_identity=text_cache.identity, device=device)
    opened = g7.load_checkpoint_for_inference(Path(REMOTE_RUNS_ROOT) / checkpoint_relative, model)
    # The checkpoint must be the one THIS variant produced. For the reference that is
    # the M9 pin; for every other row it is that row's own architecture identity, so
    # a B08 checkpoint can never be scored as though it were a baseline.
    expected_architecture = (EXPECTED_ARCHITECTURE_IDENTITY if not architecture_delta(resolved)
                             else model.architecture_identity())
    if opened["identity"].get("architecture_identity") != expected_architecture:
        raise RuntimeError(f"checkpoint architecture identity {opened['identity'].get('architecture_identity')} "
                           f"does not match this variant's {expected_architecture}")

    # Bind the FROZEN source calibration from the run that produced the checkpoint.
    # It is source-side only: fitted on `source_dev`, never on target.
    calibration_path = (Path(REMOTE_RUNS_ROOT) / checkpoint_relative).parent.parent / "calibration" / "source_dev.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_hash = calibration_hash or str(calibration.get("calibration_hash", ""))
        calibration_sha256 = calibration_sha256 or _sha256(calibration_path)
        if temperature is None: temperature = float(calibration["temperature"])
        if threshold == 0.5: threshold = float(calibration["selected_threshold"])

    lock = json.loads((Path(REMOTE_TARGET_FEATURES) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("content_identity_sha256") != EXPECTED_TARGET_FEATURE_IDENTITY:
        raise RuntimeError("refusing to predict against an unverified target feature package")

    loader_config = load_loader_config(project / "configs/data/loader_m10_target.yaml")
    # Deliberately cover both the 4-frame and the 3-frame video cases. The selection
    # is LABEL-FREE: it comes from counting feature-manifest rows per opaque video
    # id, which says nothing about live/spoof or attack family.
    selected: list[str] | None = None
    if cover_short_videos or cover_full_videos:
        import pyarrow.parquet as pq
        from collections import Counter
        manifest = pq.read_table(Path(REMOTE_TARGET_FEATURES) / "manifests" / "target_test_features.parquet")
        per_video = Counter(manifest.column("source_record_id").to_pylist())
        short = sorted(video for video, count in per_video.items() if count < 4)[:cover_short_videos]
        full = sorted(video for video, count in per_video.items() if count == 4)[:cover_full_videos]
        selected = short + full
    batches = g7.target_batches(Path(REMOTE_TARGET_FEATURES), loader_config,
                                cache_root=Path(REMOTE_M10_CACHE), limit=limit,
                                batch_size=batch_size, firewall=None, video_ids=selected)
    inference_hash = g7.inference_config_hash(
        variant=variant, flags=resolved.flags(), threshold=threshold,
        unknown_threshold=None, temperature=temperature,
        package_identity=EXPECTED_TARGET_FEATURE_IDENTITY,
        architecture_identity=model.architecture_identity())
    rows = g7.predict_target(model, batches,
                             capabilities=g7.VariantCapabilities.from_variant(resolved),
                             threshold=threshold, unknown_threshold=None, temperature=temperature,
                             checkpoint_hash=opened["checkpoint_sha256"],
                             calibration_hash=calibration_hash,
                             inference_config_hash=inference_hash, variant=variant, device=device)
    prediction_lock = g7.build_prediction_lock(
        experiment_id=experiment_id, variant=variant, seed=seed, rows=rows,
        checkpoint_sha256=opened["checkpoint_sha256"], source_calibration_sha256=calibration_sha256,
        calibration_hash=calibration_hash, inference_config_hash=inference_hash,
        target_feature_package_identity=EXPECTED_TARGET_FEATURE_IDENTITY,
        target_package_id=EXPECTED_TARGET_PACKAGE_ID, threshold=threshold, unknown_threshold=None,
        engineering_smoke=engineering_smoke)
    destination = Path(REMOTE_RUNS_ROOT) / "m10_predictions" / experiment_id
    g7.write_predictions(destination / g7.PREDICTION_FILE, rows, variant=variant)
    g7.write_prediction_lock(destination / g7.PREDICTION_LOCK_FILE, prediction_lock)
    runs_volume.commit()

    from collections import Counter
    per_video = Counter(row["video_id"] for row in rows)
    grouping = Counter(per_video.values())
    return {"experiment_id": experiment_id, "device": device, "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
            "rows": len(rows), "videos": len(per_video),
            "frames_per_video_distribution": {str(k): v for k, v in sorted(grouping.items())},
            "checkpoint": opened, "inference_config_hash": inference_hash,
            "prediction_lock_identity": prediction_lock["prediction_lock_identity"],
            "prediction_logical_identity": prediction_lock["prediction_logical_identity"],
            "engineering_smoke": prediction_lock["engineering_smoke"],
            "scientific_use": prediction_lock["scientific_use"],
            "prompt_status": sorted({row["prompt_status"] for row in rows}),
            "region_status": sorted({row["region_status"] for row in rows}),
            "variant_flags": resolved.flags(), "variant_identity": resolved.identity(),
            "firewall": evidence, "target_labels_opened": False,
            "target_metrics_computed": False,
            "written": [str(destination / g7.PREDICTION_FILE),
                        str(destination / g7.PREDICTION_LOCK_FILE)]}


# --- bounded source-only matrix training ----------------------------------------

def resolve_matrix_row(experiment_id: str, *, expected_scientific_config_hash: str = "") -> dict:
    """Re-plan the frozen matrix INSIDE the container and return one row.

    The row's flags are not passed in over the wire: they are re-derived from
    `configs/experiments/m10_matrix.yaml`, which travels with the image, and the
    caller's `scientific_config_hash` is verified against the re-derived one. A
    launcher can therefore never hand a container a flag set the frozen matrix does
    not contain, and a drifting image can never silently train a different variant.
    """
    from prism_fas.detector.variant import variant_from_row
    from prism_fas.evaluation.experiment_matrix import build_plan
    plan = build_plan(Path("/root/project/configs/experiments/m10_matrix.yaml"))
    if plan["m10_matrix_identity"] != EXPECTED_MATRIX_IDENTITY:
        raise RuntimeError(f"matrix identity {plan['m10_matrix_identity']} != the frozen pin")
    matched = [row for row in plan["rows"] if row["experiment_id"] == experiment_id]
    if not matched: raise RuntimeError(f"{experiment_id!r} is not a row of the frozen matrix")
    row = matched[0]
    if row["status"] == "BLOCKED":
        raise RuntimeError(f"{experiment_id} is BLOCKED: {row['blocked_reason']}")
    if expected_scientific_config_hash and row["scientific_config_hash"] != expected_scientific_config_hash:
        raise RuntimeError(f"{experiment_id}: scientific config hash mismatch; the launcher expected "
                           f"{expected_scientific_config_hash} and the frozen matrix says "
                           f"{row['scientific_config_hash']}")
    return {"row": row, "variant": variant_from_row(row),
            "m10_matrix_identity": plan["m10_matrix_identity"]}


def bank_root_for(variant) -> tuple[str, str | None, str | None]:
    """The synthetic bank this variant's `recipe_conditioning` declares.

    `structured` reads the frozen M8 v3 bank. `random_operators` reads the M10
    random-operator bank — a separate, source-only, separately identified artifact
    built from the SAME live targets, the SAME operator implementations, the SAME
    routes and the SAME frozen quality thresholds, differing only in how the
    operator composition was chosen. Neither is ever a fallback for the other.
    """
    if variant.recipe_source == "m10_random_operator_bank":
        return REMOTE_RANDOM_BANK, EXPECTED_RANDOM_BANK_IDENTITY, EXPECTED_RANDOM_BANK_ID
    return REMOTE_BANK, None, None


@app.function(image=image, gpu=DEFAULT_GPU, volumes=TRAIN_VOLUMES, timeout=24 * 3600)
def m10_train_row(experiment_id: str, seed: int, run_id: str, resume: bool = True,
                  max_epochs: int | None = None, scientific_config_hash: str = "",
                  engineering_smoke: bool = False) -> dict:
    """One matrix row, source-only, under that row's OWN resolved variant.

    `source_train` optimizes; `source_dev` selects the checkpoint and fits the
    calibration and produces no gradient. The target feature package is NOT opened
    by this function and target labels are nowhere on any mounted volume.

    Every scientific switch comes from the frozen matrix row, so a baseline is a
    configuration of the shared implementation and never a forked model. The stages
    that run are the stages the variant declares: G2 exists exactly where prototypes
    do.
    """
    import time
    import torch
    from prism_fas.detector.config import load_m9_configs
    from prism_fas.detector.trainer import M9Trainer, source_isolation_report

    project = Path("/root/project")
    started = time.time()
    resolved = resolve_matrix_row(experiment_id,
                                  expected_scientific_config_hash=scientific_config_hash)
    variant, row = resolved["variant"], resolved["row"]
    if int(seed) != int(row["seed"]):
        raise RuntimeError(f"{experiment_id} declares seed {row['seed']}, the launcher asked for {seed}")
    variant.require_executable()
    configs = load_m9_configs(project / "configs/models/m9_detector.yaml",
                              project / "configs/train/m9_reference.yaml", variant)
    from dataclasses import replace
    bank_root, bank_identity, bank_id = bank_root_for(variant)
    training_config = replace(configs["training_config"], run_id=run_id, seed=int(seed),
                              # Binds the A02 control artifact into the TRAINING config
                              # identity. Empty for every structured row, so B08's config
                              # hash is unchanged; set for A02, so its checkpoint can
                              # never be resumed against the structured bank.
                              synthetic_bank_identity=bank_identity or "")
    if max_epochs is not None: training_config = replace(training_config, mixed_epochs=int(max_epochs))
    trainer = M9Trainer(
        config=training_config, detector_config=configs["detector_config"],
        package_root=Path(REMOTE_PACKAGE), bank_root=Path(bank_root),
        bank_identity=bank_identity, bank_id=bank_id,
        recipe_bank_root=project / "assets/recipe_banks/prism_recipe_bank_m7_v1",
        run_root=Path(REMOTE_RUNS_ROOT) / run_id, cache_root=Path(REMOTE_CACHE_ROOT),
        weight_root=Path(REMOTE_WEIGHT_ROOT),
        loader_config_path=project / "configs/data/loader_m4.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu",
        text_cache_path=Path(REMOTE_WEIGHT_ROOT) / "recipe_text_cache.npz",
        progress=lambda payload: print(json.dumps({"progress": payload})))
    if resume and (Path(REMOTE_RUNS_ROOT) / run_id / "checkpoints" / "last.pt").is_file():
        trainer.resume()
    for stage in trainer.stages:
        getattr(trainer, f"run_{stage.lower()}")()
        runs_volume.commit()
    # Close the run. Without this a finished run reports RUNNING forever and the
    # acceptance report would describe completed work as still in flight.
    closure = trainer.finish()
    summary = trainer.run_summary()
    runs_volume.commit()
    return {"experiment_id": experiment_id, "run_id": run_id, "seed": int(seed),
            "scientific_config_hash": row["scientific_config_hash"],
            "m10_matrix_identity": resolved["m10_matrix_identity"],
            "variant_identity": variant.identity(), "flags": variant.flags(),
            "declared_stages": list(trainer.stages), "closure": closure,
            "synthetic_bank_root": bank_root,
            "synthetic_bank_identity": bank_identity or EXPECTED_BANK_IDENTITY,
            "synthetic_bank_id": bank_id or "prism_synthetic_bank_m8_v3_e84c78cd2a9b",
            "elapsed_seconds": round(time.time() - started, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "summary": summary,
            "isolation": source_isolation_report(trainer, source_dev_opened=True),
            "engineering_smoke": bool(engineering_smoke),
            "scientific_use": not bool(engineering_smoke),
            "target_features_opened": False, "target_labels_opened": False}


@app.function(image=image, gpu=DEFAULT_GPU, volumes=TRAIN_VOLUMES, timeout=3 * 3600)
def m10_smoke_row(experiment_id: str, steps: int = 2, validation_limit: int = 96) -> dict:
    """A bounded REAL-DATA verification of one row's semantic family.

    Engineering only. It runs a handful of optimizer steps per declared stage on the
    real source package and the real bank, on an L4, to prove that the variant's
    architecture, batch composition, loss graph and stage flow survive contact with
    real tensors — not to produce a result. Its run id lives in a separate
    `m10_smokes/` namespace, it never enters the experiment registry, and it is
    labelled `engineering_smoke: true` in its own return value.
    """
    import time
    import torch
    from prism_fas.detector.config import load_m9_configs
    from prism_fas.detector.trainer import M9Trainer, source_isolation_report

    project = Path("/root/project")
    started = time.time()
    resolved = resolve_matrix_row(experiment_id)
    variant, row = resolved["variant"], resolved["row"]
    variant.require_executable()
    configs = load_m9_configs(project / "configs/models/m9_detector.yaml",
                              project / "configs/train/m9_reference.yaml", variant)
    from dataclasses import replace
    run_id = f"m10_smokes/{experiment_id}"
    training_config = replace(configs["training_config"], run_id=run_id,
                              seed=int(row["seed"]), validation_limit=int(validation_limit))
    bank_root, bank_identity, bank_id = bank_root_for(variant)
    trainer = M9Trainer(
        config=training_config, detector_config=configs["detector_config"],
        package_root=Path(REMOTE_PACKAGE), bank_root=Path(bank_root),
        bank_identity=bank_identity, bank_id=bank_id,
        recipe_bank_root=project / "assets/recipe_banks/prism_recipe_bank_m7_v1",
        run_root=Path(REMOTE_RUNS_ROOT) / run_id, cache_root=Path(REMOTE_CACHE_ROOT),
        weight_root=Path(REMOTE_WEIGHT_ROOT),
        loader_config_path=project / "configs/data/loader_m4.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu",
        validation_limit=int(validation_limit),
        text_cache_path=Path(REMOTE_WEIGHT_ROOT) / "recipe_text_cache.npz")

    evidence: dict = {}
    trainer.enter_stage("G1")
    g1 = trainer.run_epoch("G1", max_steps=int(steps))
    evidence["G1"] = {"steps": len(g1), "records": g1[:2]}
    trainer.complete_stage("G1", {"steps": len(g1), "global_step": trainer.global_step})
    if variant.has_manifold:
        g2 = trainer.run_g2_smoke()
        evidence["G2"] = {"prototype_identity_sha256": g2["prototype_identity_sha256"],
                          "manifold_sites": list(trainer.model.manifold.region_names),
                          "k": trainer.model.manifold.k,
                          "distance_scale": trainer.model.distance_scale.detach().cpu().tolist()}
    trainer.enter_stage("G5")
    trainer.step_in_epoch = 0
    g5 = trainer.run_epoch("G5", start_step=0, max_steps=int(steps))
    evidence["G5"] = {"steps": len(g5), "records": g5[:2]}
    trainer.complete_stage("G5", {"steps": len(g5), "global_step": trainer.global_step,
                                  "checkpoint_sha256": trainer.save("last")})
    evidence["G6"] = trainer.run_g6()["thresholds"]
    runs_volume.commit()

    records = g1 + g5
    import math
    return {"experiment_id": experiment_id, "run_id": run_id,
            "engineering_smoke": True, "scientific_use": False,
            "variant_flags": variant.flags(), "variant_identity": variant.identity(),
            "declared_stages": list(trainer.stages),
            "architecture_identity": trainer.model.architecture_identity(),
            "parameter_counts": trainer.model.parameter_counts(),
            "optimizer_groups": [{"name": str(group.get("name")),
                                  "parameters": sum(p.numel() for p in group["params"])}
                                 for group in trainer.optimizer.param_groups],
            "batch_contract": trainer.samplers["G5"].contract.payload(),
            "dataset": trainer.dataset.summary(),
            "active_loss_terms": {stage: trainer.variant.stage_loss_terms(stage)
                                  for stage in trainer.stages if stage != "G6"},
            "all_losses_finite": all(math.isfinite(record["L_total"]) for record in records),
            "composition_seen": [record["composition"] for record in records[:2]],
            "evidence": evidence,
            "elapsed_seconds": round(time.time() - started, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "isolation": source_isolation_report(trainer, source_dev_opened=True),
            "target_features_opened": False, "target_labels_opened": False}


@app.function(image=image, gpu=DEFAULT_GPU, volumes=TRAIN_VOLUMES, timeout=3600)
def m10_parity_probe(experiment_id: str = "B08-s20260806", steps: int = 5,
                     amp: bool = False,
                     shared_checkpoint: str = "m9_reference_seed20260806/checkpoints/best.pt") -> dict:
    """The Modal half of the A09 bounded step-parity protocol.

    A handful of optimizer steps at the row's own configuration and seed, recording
    the batch, the loss trajectory and the forward outputs. It trains nothing to
    completion, selects no checkpoint and writes no calibration — it exists only to
    be compared against the local half.
    """
    import torch
    from prism_fas.detector.config import load_m9_configs
    from prism_fas.detector.trainer import M9Trainer
    from prism_fas.evaluation.backend_parity import probe_payload

    project = Path("/root/project")
    resolved = resolve_matrix_row(experiment_id)
    variant = resolved["variant"].require_executable()
    configs = load_m9_configs(project / "configs/models/m9_detector.yaml",
                              project / "configs/train/m9_reference.yaml", variant)
    from dataclasses import replace
    # `configs/cloud/modal_m6.yaml` declares `parity: precision: fp32`, and the
    # tolerances beside it were fitted for fp32-vs-fp32. Leaving AMP on here would
    # compare an L4 running bf16 against a CPU running fp32, which exceeds those
    # tolerances by three orders of magnitude for reasons that have nothing to do
    # with the backend question the row asks. AMP therefore defaults OFF for the
    # probe; it is a parameter so the AMP-vs-fp32 gap can still be measured and
    # reported separately, rather than silently conflated with backend parity.
    training_config = replace(configs["training_config"],
                              run_id=f"m10_parity/{experiment_id}", seed=int(resolved["row"]["seed"]),
                              amp=bool(amp))
    bank_root, bank_identity, bank_id = bank_root_for(variant)
    trainer = M9Trainer(
        config=training_config, detector_config=configs["detector_config"],
        package_root=Path(REMOTE_PACKAGE), bank_root=Path(bank_root),
        bank_identity=bank_identity, bank_id=bank_id,
        recipe_bank_root=project / "assets/recipe_banks/prism_recipe_bank_m7_v1",
        run_root=Path(REMOTE_RUNS_ROOT) / "m10_parity" / experiment_id,
        cache_root=Path(REMOTE_CACHE_ROOT), weight_root=Path(REMOTE_WEIGHT_ROOT),
        loader_config_path=project / "configs/data/loader_m4.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu",
        text_cache_path=Path(REMOTE_WEIGHT_ROOT) / "recipe_text_cache.npz")
    device_report = {}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        device_report = {"gpu_name": properties.name, "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda}
    payload = probe_payload(trainer, backend="modal", steps=int(steps), device_report=device_report,
                            shared_checkpoint=(Path(REMOTE_RUNS_ROOT) / shared_checkpoint
                                               if shared_checkpoint else None))
    return {"experiment_id": experiment_id, **payload}


@app.function(image=image, volumes=TRAIN_VOLUMES, timeout=3600, cpu=2.0, memory=8192)
def m10_collect_runs(experiment_ids: list[str]) -> dict:
    """Read each row's run artifacts from the runs volume and return the evidence
    the registry needs. Source-side only; no target artifact is touched.

    Reading rather than trusting: the status, the selected checkpoint SHA, the
    `source_dev` metrics and the calibration are taken from what the run actually
    wrote, so a row cannot be recorded COMPLETED because a launcher said so.
    """
    import hashlib
    collected: dict = {}
    for experiment_id in experiment_ids:
        root = Path(REMOTE_RUNS_ROOT) / experiment_id
        entry: dict = {"experiment_id": experiment_id, "run_root_exists": root.is_dir()}
        run_path = root / "run.json"
        if run_path.is_file():
            summary = json.loads(run_path.read_text(encoding="utf-8"))
            entry["status"] = summary.get("status")
            entry["stage_lineage"] = [{"stage": item["stage"], "status": item.get("status")}
                                      for item in summary.get("stage_lineage") or []]
            entry["best_metrics"] = summary.get("best_metrics")
            entry["identity"] = summary.get("identity")
            entry["variant"] = summary.get("variant", {}).get("flags")
            entry["variant_identity"] = summary.get("variant", {}).get("variant_identity_sha256")
            entry["declared_stages"] = summary.get("declared_stages")
            entry["global_step"] = summary.get("global_step")
            entry["epoch"] = summary.get("epoch")
            entry["parameter_counts"] = summary.get("parameter_counts")
            entry["optimizer_groups"] = summary.get("optimizer_groups")
            entry["dataset"] = summary.get("dataset")
            entry["ema_enabled"] = summary.get("ema_enabled")
            entry["target_test_opened"] = summary.get("target_test_opened")
        best = root / "checkpoints" / "best.pt"
        if best.is_file():
            entry["best_checkpoint_path"] = f"{experiment_id}/checkpoints/best.pt"
            entry["best_checkpoint_sha256"] = _sha256(best)
            entry["best_checkpoint_bytes"] = best.stat().st_size
        calibration = root / "calibration" / "source_dev.json"
        if calibration.is_file():
            payload = json.loads(calibration.read_text(encoding="utf-8"))
            entry["source_calibration_sha256"] = _sha256(calibration)
            entry["calibration_hash"] = payload.get("calibration_hash")
            entry["calibration"] = {key: payload.get(key) for key in
                                    ("temperature", "selected_threshold", "criterion",
                                     "reject_policy", "checkpoint_sha256")}
        for stage in ("G1", "G2", "G5", "G6"):
            marker = root / "stages" / stage / "output_hashes.json"
            if marker.is_file():
                entry.setdefault("stage_outputs", {})[stage] = json.loads(
                    marker.read_text(encoding="utf-8"))
        collected[experiment_id] = entry
    return {"collected": collected, "count": len(collected), "target_labels_opened": False}


@app.local_entrypoint()
def main(action: str = "verify", experiment_id: str = "g7_new_package_smoke",
         checkpoint_relative: str = "", limit: int | None = 320, seed: int = 20260806,
         run_id: str = "", threshold: float = 0.5, temperature: float | None = None,
         max_epochs: int | None = None, scientific_config_hash: str = "",
         engineering_smoke: bool = False, variant: str = "B08", concurrency: int = 3) -> None:
    if action == "verify":
        print(json.dumps(m10_verify_target_features.remote(), indent=1, default=str))
    elif action == "g7":
        print(json.dumps(m10_g7_predict.remote(experiment_id, checkpoint_relative, variant=variant,
                                               seed=seed, limit=limit,
                                               threshold=threshold, temperature=temperature,
                                               engineering_smoke=engineering_smoke,
                                               cover_short_videos=24, cover_full_videos=76),
                         indent=1, default=str))
    elif action == "train":
        print(json.dumps(m10_train_row.remote(experiment_id, seed, run_id or experiment_id,
                                              max_epochs=max_epochs,
                                              scientific_config_hash=scientific_config_hash,
                                              engineering_smoke=engineering_smoke),
                         indent=1, default=str))
    elif action == "parity-probe":
        # `--amp` opts INTO mixed precision; the declared protocol is fp32.
        print(json.dumps(m10_parity_probe.remote(experiment_id, 5, engineering_smoke),
                         indent=1, default=str))
    elif action == "collect":
        ids = [value.strip() for value in experiment_id.split(",") if value.strip()]
        print(json.dumps(m10_collect_runs.remote(ids), indent=1, default=str))
    elif action == "matrix":
        # The source-only scientific matrix, in bounded chunks. One unique run id per
        # row (the experiment id itself), so a relaunch resumes rather than
        # duplicating, and the workspace GPU quota is respected rather than exceeded.
        ids = [value.strip() for value in experiment_id.split(",") if value.strip()]
        width = max(1, int(concurrency))
        results: list = []
        for start in range(0, len(ids), width):
            chunk = ids[start:start + width]
            print(json.dumps({"launching": chunk, "done": start, "total": len(ids)}))
            seeds = [int(name.rsplit("-s", 1)[1]) for name in chunk]
            results += list(m10_train_row.starmap(
                [(name, seed, name) for name, seed in zip(chunk, seeds)],
                kwargs={"max_epochs": max_epochs}))
            for entry in results[-len(chunk):]:
                print(json.dumps({"finished": entry["experiment_id"],
                                  "status": entry["closure"]["status"],
                                  "seconds": entry["elapsed_seconds"],
                                  "best": entry["summary"]["best_metrics"]}, default=str))
        print(json.dumps({"matrix": results}, indent=1, default=str))
    elif action == "smoke":
        # Comma-separated ids so one invocation can cover several semantic families,
        # fanned out in bounded chunks rather than all at once: the workspace GPU
        # quota is a real limit and exceeding it deliberately is not an option.
        ids = [value.strip() for value in experiment_id.split(",") if value.strip()]
        width = max(1, int(concurrency))
        results: list = []
        for start in range(0, len(ids), width):
            chunk = ids[start:start + width]
            print(json.dumps({"launching": chunk}))
            results += list(m10_smoke_row.map(chunk, kwargs={"steps": 2}))
        print(json.dumps({"smokes": results}, indent=1, default=str))
    else:
        raise SystemExit(f"unknown action {action!r}; use verify | g7 | train | smoke")
