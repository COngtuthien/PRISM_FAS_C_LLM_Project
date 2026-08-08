"""The ONE typed, validated resolved experiment variant.

`configs/experiments/m10_matrix.yaml` declares each Table 59 baseline and Table 60
ablation as a flag set. This module turns a flag set into a single frozen, validated
object that then drives EVERYTHING: the detector architecture, the batch contract,
the dataset construction, the sampler, the loss graph, the required stages, the
architecture identity, the training identity, the checkpoint metadata and the G7
inference adapter.

Two rules this module exists to enforce:

1.  **No experiment-id branching.** Nothing anywhere in the trainer or the model may
    ask "am I B03?". It asks the variant what it HAS. `if experiment_id == "B03"` is
    the failure mode this object was written to make impossible.
2.  **Fail closed.** An unknown flag value, an unknown combination or a
    contradictory combination raises. A variant that cannot be built is never
    silently repaired into a neighbouring one, and a flag that would change nothing
    about behaviour is rejected rather than accepted as decoration.

Every scientifically meaningful switch enters an identity: `architecture_payload()`
for the switches that change the tensor graph, `training_payload()` for the switches
that change what is optimized rather than what is instantiated. A checkpoint trained
with `prompt=off` therefore cannot be loaded as `prompt=frozen_prompt`.

This module never imports modal, never imports the evaluation package and never
mentions a target path.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, replace
from typing import Any

VARIANT_SCHEMA_VERSION = "m10-resolved-variant-v1"

# --- the frozen flag vocabulary ---------------------------------------------
# Identical to `configs/experiments/m10_matrix.yaml: flag_vocabulary`. A test
# asserts the two agree, so the YAML can never declare a value the code cannot
# honour and the code can never accept a value the matrix has not declared.
FLAG_VOCABULARY: dict[str, tuple[Any, ...]] = {
    "local_branch": ("convnext", "off"),
    "global_branch": ("siglip2_frozen", "off"),
    "fusion": ("single_logit", "simple_concat", "prism_noisy_or"),
    "region": ("on", "off"),
    "manifold": ("off", "global_center", "multi_prototype"),
    "prototype_k": (0, 1, 2, 4, 6),
    "synthetic": ("none", "physics_only", "gpat_only", "bank_physics_gpat"),
    "recipe_conditioning": ("structured", "random_operators", "off"),
    "quality_weighting": ("q_weighted", "hard_gate_only", "off"),
    "outlier_loss": ("off", "image_level", "mask_aware"),
    "prompt": ("off", "frozen_prompt", "adapter"),
    "sampler": ("domain_class_balanced", "naive_concat"),
    "frames_per_video": (4, 16, 32, 64),
}
FLAG_KEYS = tuple(FLAG_VOCABULARY)

# The B08 reference flag set, identical to `reference_flags` in the matrix config.
REFERENCE_FLAGS: dict[str, Any] = {
    "local_branch": "convnext", "global_branch": "siglip2_frozen", "fusion": "prism_noisy_or",
    "region": "on", "manifold": "multi_prototype", "prototype_k": 4,
    "synthetic": "bank_physics_gpat", "recipe_conditioning": "structured",
    "quality_weighting": "q_weighted", "outlier_loss": "mask_aware", "prompt": "frozen_prompt",
    "sampler": "domain_class_balanced", "frames_per_video": 4}

# The only frame density the frozen source and target packages exist at. The three
# denser A10 rows are BLOCKED in the matrix for exactly this reason; resolving one
# here would silently pretend data exists that does not.
SUPPORTED_FRAMES_PER_VIDEO = 4

ROUTES_BY_FLAG: dict[str, tuple[str, ...]] = {
    "none": (), "physics_only": ("physics",), "gpat_only": ("gpat",),
    "bank_physics_gpat": ("physics", "gpat")}

# Every declared loss term, in the frozen M9 order.
LOSS_TERMS = ("L_cls_real", "L_cls_syn", "L_local", "L_MIL", "L_real", "L_out",
              "L_clean", "L_prompt", "L_cons", "L_risk")


class VariantError(ValueError):
    """A flag set is unknown, incomplete or self-contradictory."""


def _stable(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedExperimentVariant:
    """One fully resolved, validated scientific configuration.

    Construct with `ResolvedExperimentVariant.resolve(flags)`; the bare constructor
    is deliberately unvalidated so `dataclasses.replace` stays cheap, and every
    public entry point calls `validate()`.
    """
    local_branch: str = REFERENCE_FLAGS["local_branch"]
    global_branch: str = REFERENCE_FLAGS["global_branch"]
    fusion: str = REFERENCE_FLAGS["fusion"]
    region: str = REFERENCE_FLAGS["region"]
    manifold: str = REFERENCE_FLAGS["manifold"]
    prototype_k: int = REFERENCE_FLAGS["prototype_k"]
    synthetic: str = REFERENCE_FLAGS["synthetic"]
    recipe_conditioning: str = REFERENCE_FLAGS["recipe_conditioning"]
    quality_weighting: str = REFERENCE_FLAGS["quality_weighting"]
    outlier_loss: str = REFERENCE_FLAGS["outlier_loss"]
    prompt: str = REFERENCE_FLAGS["prompt"]
    sampler: str = REFERENCE_FLAGS["sampler"]
    frames_per_video: int = REFERENCE_FLAGS["frames_per_video"]

    # --- construction -------------------------------------------------------
    @classmethod
    def reference(cls) -> "ResolvedExperimentVariant":
        """The frozen B08 configuration."""
        return cls().validate()

    @classmethod
    def resolve(cls, flags: dict[str, Any] | None = None, *,
                base: dict[str, Any] | None = None) -> "ResolvedExperimentVariant":
        """Resolve a flag DELTA against the B08 reference, then validate.

        Every matrix row is expressed as a delta, so resolving one is the same
        operation the planner performs — which is why the two agree by
        construction rather than by convention.
        """
        resolved = {**REFERENCE_FLAGS, **(base or {}), **{str(k): v for k, v in (flags or {}).items()}}
        unknown = sorted(set(resolved) - set(FLAG_KEYS))
        if unknown: raise VariantError(f"unknown flags {unknown}")
        return cls(**{key: resolved[key] for key in FLAG_KEYS}).validate()

    def with_flags(self, **flags: Any) -> "ResolvedExperimentVariant":
        return replace(self, **flags).validate()

    # --- validation ---------------------------------------------------------
    def flags(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in FLAG_KEYS}

    def validate(self) -> "ResolvedExperimentVariant":
        """Every vocabulary and consistency rule. A breach raises; nothing is repaired."""
        for key, allowed in FLAG_VOCABULARY.items():
            value = getattr(self, key)
            if value not in allowed:
                raise VariantError(f"flag {key}={value!r} is outside the frozen vocabulary {list(allowed)}")
        problems = self._contradictions()
        if problems: raise VariantError(f"contradictory flags: {problems}")
        return self

    def _contradictions(self) -> list[str]:
        """The declared consistency rules, in one place.

        `evaluation.experiment_matrix` validates matrix rows against this same list,
        so a row the matrix accepts is always a variant this module can build.
        """
        k, branches = int(self.prototype_k), self.branch_count
        problems: list[str] = []
        if (self.manifold == "off") != (k == 0):
            problems.append("manifold=off must pair with prototype_k=0")
        if self.manifold == "global_center" and k != 1:
            problems.append("a global center is exactly one prototype")
        if self.manifold == "multi_prototype" and (k < 1 or self.region != "on"):
            problems.append("multi-prototype manifolds are regional and need at least one prototype")
        if self.region == "on" and self.fusion != "prism_noisy_or":
            problems.append("the regional detector uses the Table 34 fusion")
        if self.region == "off" and self.fusion == "prism_noisy_or":
            problems.append("the Table 34 fusion needs the regional evidence term")
        if self.prompt != "off" and (self.region != "on" or self.global_branch == "off"):
            problems.append("PromptHead is defined over region embeddings and the frozen text tower")
        if self.outlier_loss == "mask_aware" and self.region != "on":
            problems.append("a mask-aware outlier loss needs region masks")
        # A mask-aware term is defined per REGION against the regional manifold. With
        # a single global center there is one distance and no per-region mask to be
        # aware of, so the pairing is only coherent when no synthetic sample exists
        # for the term to apply to at all.
        if (self.outlier_loss == "mask_aware" and self.manifold != "multi_prototype"
                and self.synthetic != "none"):
            problems.append("a mask-aware outlier loss over synthetic data needs the regional manifold")
        if self.synthetic == "none" and self.quality_weighting != "off":
            problems.append("q weights synthetic samples; with no synthetic data there is nothing to weight")
        if self.recipe_conditioning != "off" and self.synthetic == "none":
            problems.append("recipe conditioning requires synthetic samples")
        if self.fusion == "single_logit" and branches != 1:
            problems.append("a single-logit baseline has exactly one backbone branch")
        if self.fusion in ("simple_concat", "prism_noisy_or") and branches != 2:
            problems.append(f"{self.fusion} needs both backbone branches")
        if branches == 0:
            problems.append("a detector needs at least one backbone branch")
        return problems

    # --- capabilities: what this variant HAS ---------------------------------
    @property
    def has_local(self) -> bool: return self.local_branch != "off"
    @property
    def has_global(self) -> bool: return self.global_branch != "off"
    @property
    def branch_count(self) -> int: return int(self.has_local) + int(self.has_global)
    @property
    def has_region_path(self) -> bool:
        """The semantic RegionQuery / cross-attention / RegionPool path."""
        return self.region == "on"
    @property
    def has_manifold(self) -> bool: return self.manifold != "off"
    @property
    def manifold_scope(self) -> str:
        """`none` | `global` (one center over a pooled embedding) | `regional`."""
        return {"off": "none", "global_center": "global", "multi_prototype": "regional"}[self.manifold]
    @property
    def manifold_slots(self) -> int:
        """How many manifold sites exist: 0, 1 (global center) or 9 (regional)."""
        from .contracts import REGION_COUNT
        return {"none": 0, "global": 1, "regional": REGION_COUNT}[self.manifold_scope]
    @property
    def has_prompt(self) -> bool: return self.prompt != "off"
    @property
    def prompt_adapter_trainable(self) -> bool: return self.prompt == "adapter"
    @property
    def uses_synthetic(self) -> bool: return self.synthetic != "none"
    @property
    def synthetic_routes(self) -> tuple[str, ...]: return ROUTES_BY_FLAG[self.synthetic]
    @property
    def requires_both_routes(self) -> bool: return len(self.synthetic_routes) == 2
    @property
    def uses_quality_weight(self) -> bool:
        """`q_weighted` uses the M8 soft weight; `hard_gate_only` gives every ACCEPTED
        sample equal weight; `off` has no synthetic bracket to weight at all."""
        return self.quality_weighting == "q_weighted"
    @property
    def recipe_source(self) -> str:
        """Which recipe artifact the synthetic samples were composed from."""
        return {"structured": "m7_structured_bank", "random_operators": "m10_random_operator_bank",
                "off": "none"}[self.recipe_conditioning]
    @property
    def consumes_recipe_identity(self) -> bool:
        """Whether the recipe id / artifact family reaches the loss graph at all."""
        return self.recipe_conditioning != "off"
    @property
    def domain_balance(self) -> bool: return self.sampler == "domain_class_balanced"
    @property
    def fuses_region_evidence(self) -> bool:
        """Whether `s_region` enters `s_final`.

        Only the Table 34 noisy-or fusion consumes the regional evidence term. Under
        `simple_concat` the manifold is a TRAINING objective (`L_real` / `L_out`) and
        the score comes from the concat classifier — which is why B05's G7 rows write
        `s_region = null` rather than a value the fusion never used.
        """
        return self.fusion == "prism_noisy_or" and self.has_manifold
    @property
    def fuses_prompt_evidence(self) -> bool:
        return self.fusion == "prism_noisy_or" and self.has_prompt

    # --- derived plans -------------------------------------------------------
    def required_stages(self) -> tuple[str, ...]:
        """G2 exists exactly where prototypes exist. Nothing is run for its own sake."""
        return ("G1", "G2", "G5", "G6") if self.has_manifold else ("G1", "G5", "G6")

    def active_loss_terms(self) -> dict[str, bool]:
        """The loss graph, derived from active capabilities.

        An inactive term is STRUCTURALLY absent — never computed and never evaluated
        as an arbitrary constant that would then be logged as a measurement.
        """
        synthetic, manifold = self.uses_synthetic, self.has_manifold
        return {
            "L_cls_real": True,
            "L_cls_syn": synthetic,
            # Both local-token terms need the ConvNeXt token grid.
            "L_local": synthetic and self.has_local,
            "L_MIL": self.has_local,
            "L_real": manifold,
            "L_out": synthetic and manifold and self.outlier_loss != "off",
            # `L_clean` is the mask-aware companion: it is defined by `1 - m_r`, so an
            # image-level outlier term has no clean counterpart to compute.
            "L_clean": synthetic and manifold and self.outlier_loss == "mask_aware",
            "L_prompt": synthetic and self.has_prompt and self.has_region_path,
            # Consistency compares the two fused evidence terms; without the noisy-or
            # fusion there is only one score and nothing to reconcile.
            "L_cons": self.fuses_region_evidence,
            "L_risk": True}

    def stage_loss_terms(self, stage: str) -> dict[str, bool]:
        """The active terms narrowed to what a STAGE can legitimately compute.

        G1 has no prototypes and no synthetic samples; G2 has prototypes but still no
        synthetic samples; G5 is the variant's full graph.
        """
        active = self.active_loss_terms()
        manifold_terms = ("L_real", "L_out", "L_clean")
        synthetic_terms = ("L_cls_syn", "L_local", "L_out", "L_clean", "L_prompt")
        if stage == "G1":
            blocked = set(manifold_terms) | set(synthetic_terms)
        elif stage == "G2":
            blocked = set(synthetic_terms)
        elif stage in ("G5", "G6"):
            blocked = set()
        else:
            raise VariantError(f"unknown stage {stage!r}")
        return {name: bool(value and name not in blocked) for name, value in active.items()}

    def batch_phase(self, stage: str) -> str:
        """`real_only` where no synthetic sample may appear, `mixed` otherwise.

        G1/G2 read "Source real live/spoof" (Table 39) for every variant. G5 is
        real-only too when the variant declares `synthetic: none` — that is not a
        warm-up shortcut, it is what B00-B07 are.
        """
        if stage in ("G1", "G2"): return "real_only"
        if stage in ("G5", "G6"): return "mixed" if self.uses_synthetic else "real_only"
        raise VariantError(f"unknown stage {stage!r}")

    def trainable_modules(self) -> tuple[str, ...]:
        """Which trainable module groups this variant instantiates.

        Used by the audit to prove no optimizer group is empty and that a branch
        switched off leaves no unused trainable parameters behind.
        """
        modules: list[str] = []
        if self.has_local: modules += ["local_backbone", "local_projection"]
        if self.has_global: modules += ["global_projection"]
        if self.has_region_path: modules += ["region_query", "region_attention", "region_pool", "region_norm"]
        if self.has_local: modules += ["local_head"]
        if self.fusion == "prism_noisy_or": modules += ["global_head"]
        else: modules += ["fusion_classifier"]
        if self.has_prompt: modules += ["prompt_head"]
        return tuple(sorted(set(modules)))

    # --- identity ------------------------------------------------------------
    def architecture_payload(self) -> dict[str, Any]:
        """The switches that change the TENSOR GRAPH.

        These enter `DetectorConfig.identity()` and therefore the checkpoint's
        architecture identity: a checkpoint built without a PromptHead cannot be
        loaded into one that has it, because the state dict and this hash both differ.
        """
        return {"variant_schema_version": VARIANT_SCHEMA_VERSION,
                "local_branch": self.local_branch, "global_branch": self.global_branch,
                "fusion": self.fusion, "region": self.region, "manifold": self.manifold,
                "prototype_k": int(self.prototype_k), "prompt": self.prompt,
                "manifold_scope": self.manifold_scope, "manifold_slots": self.manifold_slots,
                "fuses_region_evidence": self.fuses_region_evidence,
                "fuses_prompt_evidence": self.fuses_prompt_evidence}

    def training_payload(self) -> dict[str, Any]:
        """The switches that change WHAT IS OPTIMIZED rather than what is instantiated.

        `synthetic`, `recipe_conditioning`, `quality_weighting`, `outlier_loss` and
        `sampler` leave every parameter shape untouched, so they cannot be caught by
        an architecture hash. They are bound here instead, and this hash enters the
        training config identity — so a B08 checkpoint can never be resumed as A04.
        """
        return {"variant_schema_version": VARIANT_SCHEMA_VERSION,
                "synthetic": self.synthetic, "synthetic_routes": list(self.synthetic_routes),
                "recipe_conditioning": self.recipe_conditioning, "recipe_source": self.recipe_source,
                "quality_weighting": self.quality_weighting, "outlier_loss": self.outlier_loss,
                "sampler": self.sampler, "domain_balance": self.domain_balance,
                "frames_per_video": int(self.frames_per_video),
                "active_loss_terms": self.active_loss_terms(),
                "required_stages": list(self.required_stages())}

    def payload(self) -> dict[str, Any]:
        """Everything, for `resolved_config.yaml` and the run registry."""
        return {"variant_schema_version": VARIANT_SCHEMA_VERSION, "flags": self.flags(),
                "architecture": self.architecture_payload(), "training": self.training_payload(),
                "trainable_modules": list(self.trainable_modules()),
                "variant_identity_sha256": self.identity()}

    def architecture_identity(self) -> str: return _stable(self.architecture_payload())
    def training_identity(self) -> str: return _stable(self.training_payload())

    def identity(self) -> str:
        """The full scientific identity of this variant: architecture AND training."""
        return _stable({"architecture": self.architecture_payload(),
                        "training": self.training_payload()})

    # --- executability -------------------------------------------------------
    def executable(self) -> tuple[bool, str | None]:
        """Whether this variant can actually be materialized from frozen artifacts.

        The only currently inexecutable case is a frame density the frozen packages
        do not exist at, which the matrix already carries as BLOCKED rows.
        """
        if int(self.frames_per_video) != SUPPORTED_FRAMES_PER_VIDEO:
            return False, (f"frames_per_video={self.frames_per_video} is not the frozen frame plan "
                           f"({SUPPORTED_FRAMES_PER_VIDEO}); the source and target packages do not "
                           "exist at that density")
        return True, None

    def require_executable(self) -> "ResolvedExperimentVariant":
        ok, reason = self.executable()
        if not ok: raise VariantError(reason or "variant is not executable")
        return self


def variant_from_row(row: dict[str, Any]) -> ResolvedExperimentVariant:
    """Resolve a materialized matrix row's flag set. The row already carries every
    flag resolved against the reference, so this is a validation, not a merge."""
    flags = dict(row.get("flags") or {})
    missing = sorted(set(FLAG_KEYS) - set(flags))
    if missing: raise VariantError(f"{row.get('experiment_id')}: unresolved flags {missing}")
    return ResolvedExperimentVariant.resolve(flags)


def describe_difference(left: ResolvedExperimentVariant,
                        right: ResolvedExperimentVariant) -> dict[str, dict[str, Any]]:
    """Exactly which flags differ. Used by the matrix-difference audit to prove a
    row changes the dimensions it declares and no others."""
    return {key: {"left": getattr(left, key), "right": getattr(right, key)}
            for key in FLAG_KEYS if getattr(left, key) != getattr(right, key)}
