"""The autograd probe that runs before C4, and the defect it was written against.

On the Linux RTX 5090 host the preflight reached every earlier gate — bundle,
disk, write access, target firewall, GPU_SCIENTIFIC_FULL intent — and then died
with `AttributeError: 'Tensor' object has no attribute 'image'`. The probe had
invented its own input contract: it handed `PRISMDetector.forward` a bare
`torch.randn(2, 3, 224, 224)` while the trainer hands it a `DetectorBatch`.

So the first job of this file is to hold the probe to the REAL contract, and to
keep a bare tensor recognisably wrong. The second is to prove the probe still
refuses what it exists to refuse: a missing gradient, a non-finite gradient, and
work that quietly executed on the host after a CUDA device was selected.

Everything here runs on the CPU. The device rules are asserted against the pure
verifier rather than against a card, so a CPU laptop and the GPU host check the
same logic.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prism_fas.detector.contracts import DetectorBatch
from prism_fas.detector.variant import ResolvedExperimentVariant
from prism_fas.pipeline import gpu_preflight
from prism_fas.pipeline.adapters.c7 import TRACK_R_FLAGS

REPO = Path(__file__).resolve().parents[2]
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def variant() -> ResolvedExperimentVariant:
    return ResolvedExperimentVariant.resolve(TRACK_R_FLAGS)


@pytest.fixture(scope="module")
def roundtrip() -> dict:
    """One real forward/backward, shared: it is cheap on a GPU, not on a laptop."""
    return gpu_preflight._model_roundtrip(torch, CPU)


# --- the real batch contract -------------------------------------------------

def test_the_probe_builds_a_detector_batch_not_a_tensor(variant) -> None:
    batch, _ = gpu_preflight._representative_batch(variant, CPU)

    assert isinstance(batch, DetectorBatch), (
        "the probe must speak the trainer's input contract, not invent one")
    assert hasattr(batch, "image")
    batch.validate()


def test_the_probe_batch_matches_the_declared_training_composition(variant) -> None:
    """The composition comes from `batch_contract_for`, not from a guess."""
    from prism_fas.detector.trainer import M9TrainingConfig, batch_contract_for

    batch, contract = gpu_preflight._representative_batch(variant, CPU)
    declared = batch_contract_for(
        gpu_preflight.PREFLIGHT_STAGE,
        M9TrainingConfig(run_id="gpu_preflight", variant=variant, steps_per_epoch=1))

    assert contract.payload() == declared.payload()
    assert batch.batch_size == declared.batch_size
    assert set(batch.kinds()) == {"real_live", "real_spoof", "synthetic_spoof"}


def test_a_bare_tensor_is_still_rejected_by_the_detector(variant) -> None:
    """The regression itself. If this ever stops raising, the model grew a second
    input contract and the probe would no longer be exercising the real one."""
    from prism_fas.evaluation.variant_audit import build_audit_detector

    model = build_audit_detector(variant)
    with pytest.raises(AttributeError, match="image"):
        model(torch.randn(2, 3, 224, 224))


def test_the_audit_stub_tower_follows_the_image_device() -> None:
    """The defect hiding behind the first one: a stub that always draws on the
    host hands CPU tokens to a model on the GPU, and `region_embeddings` fails
    there. `meta` stands in for "not the CPU" so this holds on a CPU laptop too."""
    from prism_fas.evaluation.variant_audit import _StubTower

    tower = _StubTower(dim=8, tokens=4)
    elsewhere = tower(pixel_values=torch.zeros(2, 3, 224, 224, device="meta"))
    assert elsewhere.last_hidden_state.device.type == "meta"
    assert elsewhere.pooler_output.device.type == "meta"

    # ...while the CPU audit it was written for keeps its exact seeded values.
    host = tower(pixel_values=torch.zeros(2, 3, 224, 224))
    expected = torch.rand(2, 4, 8, generator=torch.Generator().manual_seed(20260806))
    assert torch.equal(host.last_hidden_state, expected)


# --- forward, loss, backward -------------------------------------------------

def test_forward_produces_a_finite_scalar_loss(roundtrip) -> None:
    assert roundtrip["loss_is_finite"] is True
    assert roundtrip["loss"] == roundtrip["loss"]           # not NaN
    assert roundtrip["stage"] == gpu_preflight.PREFLIGHT_STAGE
    assert roundtrip["active_loss_terms"], "the real loss graph must have run"


def test_backward_reaches_every_trainable_parameter(roundtrip) -> None:
    assert roundtrip["trainable_parameters"] > 0
    assert roundtrip["parameters_with_gradients"] == roundtrip["trainable_parameters"]


def test_every_gradient_is_finite(roundtrip) -> None:
    assert roundtrip["gradients_are_finite"] is True
    norm = roundtrip["gradient_global_norm"]
    assert norm == norm and norm > 0.0


def test_the_probe_reports_the_device_it_actually_executed_on(roundtrip) -> None:
    assert set(roundtrip["executed_on"]) >= {"batch.image", "loss", "first_gradient"}
    assert set(roundtrip["executed_on"].values()) == {"cpu"}


def test_the_probe_writes_no_artifact(roundtrip, tmp_path: Path) -> None:
    """A preflight that leaves evidence behind is a preflight that can be mistaken
    for a result. The forward/backward probe writes nothing at all."""
    before = {path for path in (REPO / "reports").rglob("*") if path.is_file()}
    gpu_preflight._model_roundtrip(torch, CPU)
    after = {path for path in (REPO / "reports").rglob("*") if path.is_file()}

    assert before == after


def test_the_probe_resolves_no_target(variant) -> None:
    """SiW-Mv2 is never opened: the batch is a seeded fixture and its declared
    datasets are source-domain only."""
    batch, _ = gpu_preflight._representative_batch(variant, CPU)

    assert all("siw" not in name.lower() for name in batch.datasets)
    assert all("siw" not in str(sample).lower() for sample in batch.sample_ids)


# --- what the probe must refuse ---------------------------------------------

def _parameter(grad: torch.Tensor | None) -> SimpleNamespace:
    return SimpleNamespace(grad=grad, requires_grad=True)


def test_a_missing_gradient_fails_the_probe() -> None:
    trainable = [("kept", _parameter(torch.ones(3))),
                 ("detached", _parameter(None))]

    with pytest.raises(gpu_preflight.GPUPreflightError) as raised:
        gpu_preflight._audit_gradients(torch, trainable)

    assert raised.value.reason == gpu_preflight.AUTOGRAD_FAILED
    assert "detached" in str(raised.value)


def test_a_non_finite_gradient_fails_the_probe() -> None:
    trainable = [("fine", _parameter(torch.ones(3))),
                 ("nan", _parameter(torch.tensor([float("nan"), 1.0])))]

    with pytest.raises(gpu_preflight.GPUPreflightError) as raised:
        gpu_preflight._audit_gradients(torch, trainable)

    assert raised.value.reason == gpu_preflight.AUTOGRAD_FAILED
    assert "nan" in str(raised.value)


def test_a_model_with_nothing_to_train_fails_the_probe() -> None:
    with pytest.raises(gpu_preflight.GPUPreflightError) as raised:
        gpu_preflight._audit_gradients(torch, [])

    assert raised.value.reason == gpu_preflight.AUTOGRAD_FAILED


def test_finite_gradients_report_the_full_count() -> None:
    detail = gpu_preflight._audit_gradients(
        torch, [("a", _parameter(torch.ones(3))), ("b", _parameter(torch.full((2,), 2.0)))])

    assert detail == {"trainable_parameters": 2, "parameters_with_gradients": 2,
                      "gradients_are_finite": True, "gradient_global_norm": 3.316625}


# --- device verification -----------------------------------------------------

def _on(device: str) -> SimpleNamespace:
    return SimpleNamespace(device=torch.device(device))


def test_a_cpu_tensor_under_a_cuda_selection_is_a_silent_fallback() -> None:
    with pytest.raises(gpu_preflight.GPUPreflightError) as raised:
        gpu_preflight._require_device(torch.device("cuda", 0),
                                      {"batch.image": _on("cpu")})

    assert raised.value.reason == gpu_preflight.AUTOGRAD_FAILED
    assert "silent CPU fallback" in str(raised.value)
    assert raised.value.detail["misplaced"] == {"batch.image": "cpu"}


def test_matching_cuda_tensors_verify() -> None:
    reported = gpu_preflight._require_device(
        torch.device("cuda", 0), {"loss": _on("cuda:0"), "grad": _on("cuda:0")})

    assert reported == {"loss": "cuda:0", "grad": "cuda:0"}


def test_the_wrong_cuda_index_is_rejected() -> None:
    with pytest.raises(gpu_preflight.GPUPreflightError):
        gpu_preflight._require_device(torch.device("cuda", 0), {"loss": _on("cuda:1")})


def test_an_unindexed_cuda_request_accepts_the_current_device() -> None:
    assert gpu_preflight._require_device(torch.device("cuda"), {"loss": _on("cuda:0")})


def test_a_none_tensor_is_not_a_device_violation() -> None:
    assert gpu_preflight._require_device(CPU, {"absent": None}) == {}
