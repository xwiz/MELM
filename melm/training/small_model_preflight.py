"""Preflight checks for checkpointed small-model stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable, Protocol

from .small_model_plan import SmallModelStageSpec, estimate_tiny_decoder_parameters


class DiskUsageLike(Protocol):
    free: int


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class SmallModelPreflightReport:
    stage_name: str
    status: str
    checks: list[PreflightCheck]
    estimated_parameters_per_arm: int
    estimated_checkpoint_bytes: int
    estimated_training_memory_lower_bound_bytes: int
    free_disk_bytes: int | None
    cuda_available: bool | None
    cuda_devices: list[dict]


def preflight_small_model_stage(
    spec: SmallModelStageSpec,
    *,
    root: str | Path = ".",
    min_free_disk_bytes: int = 2_000_000_000,
    require_cuda: bool = False,
    disk_usage: Callable[[Path], DiskUsageLike] | None = None,
    device_probe: Callable[[], dict] | None = None,
) -> SmallModelPreflightReport:
    """Check whether local machine state is plausible for a stage run."""

    root_path = Path(root)
    parameters = estimate_tiny_decoder_parameters(
        vocab_size=spec.max_vocab_size,
        sequence_length=spec.sequence_length,
        embedding_dim=spec.embedding_dim,
        layers=spec.layers,
    )
    checkpoint_bytes = parameters * 4 * len(spec.tokenizers)
    training_memory = estimate_training_memory_lower_bound(
        parameters=parameters,
        batch_size=spec.batch_size,
        sequence_length=spec.sequence_length,
        embedding_dim=spec.embedding_dim,
        layers=spec.layers,
        heads=spec.heads,
    )
    checks: list[PreflightCheck] = []

    checks.append(_path_check(root_path / spec.manifest, "manifest"))
    if spec.source_gate:
        checks.append(_path_check(root_path / spec.source_gate, "source_gate"))
    if spec.source_proxy_decision:
        checks.append(_path_check(root_path / spec.source_proxy_decision, "source_proxy_decision"))

    usage = (disk_usage or shutil.disk_usage)(root_path)
    free_disk = int(usage.free)
    required_disk = max(min_free_disk_bytes, checkpoint_bytes * 3)
    checks.append(
        PreflightCheck(
            name="disk",
            status="pass" if free_disk >= required_disk else "fail",
            detail=(
                f"free={free_disk} required={required_disk} "
                f"checkpoint_estimate={checkpoint_bytes}"
            ),
        )
    )

    probe = (device_probe or _probe_torch)()
    cuda_available = probe.get("cuda_available")
    cuda_devices = list(probe.get("devices", []))
    checks.append(
        _device_check(
            spec,
            probe,
            require_cuda=require_cuda,
            estimated_training_memory_bytes=training_memory,
        )
    )

    status = _overall_status(checks)
    return SmallModelPreflightReport(
        stage_name=spec.name,
        status=status,
        checks=checks,
        estimated_parameters_per_arm=parameters,
        estimated_checkpoint_bytes=checkpoint_bytes,
        estimated_training_memory_lower_bound_bytes=training_memory,
        free_disk_bytes=free_disk,
        cuda_available=None if cuda_available is None else bool(cuda_available),
        cuda_devices=cuda_devices,
    )


def estimate_training_memory_lower_bound(
    *,
    parameters: int,
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
    layers: int,
    heads: int,
) -> int:
    """Rough lower bound for training memory, not an OOM guarantee."""

    optimizer_and_gradients = parameters * 16
    hidden_activations = batch_size * sequence_length * embedding_dim * layers * 64
    attention_scores = batch_size * sequence_length * sequence_length * heads * layers * 4
    return optimizer_and_gradients + hidden_activations + attention_scores


def _path_check(path: Path, name: str) -> PreflightCheck:
    return PreflightCheck(
        name=name,
        status="pass" if path.exists() else "fail",
        detail=str(path),
    )


def _device_check(
    spec: SmallModelStageSpec,
    probe: dict,
    *,
    require_cuda: bool,
    estimated_training_memory_bytes: int,
) -> PreflightCheck:
    torch_available = bool(probe.get("torch_available"))
    cuda_available = bool(probe.get("cuda_available"))
    devices = list(probe.get("devices", []))
    requested_cuda = spec.device.startswith("cuda")
    if not torch_available:
        return PreflightCheck("device", "fail", "PyTorch is not importable")
    if require_cuda or requested_cuda:
        if not cuda_available:
            return PreflightCheck("device", "fail", "CUDA was required but is not available")
    if not cuda_available:
        return PreflightCheck("device", "warn", "CUDA unavailable; stage will run on CPU if launched")

    largest = max((int(device.get("total_memory", 0)) for device in devices), default=0)
    status = "pass" if largest >= estimated_training_memory_bytes * 2 else "warn"
    return PreflightCheck(
        "device",
        status,
        (
            f"cuda_available=True largest_device_memory={largest} "
            f"training_memory_lower_bound={estimated_training_memory_bytes}"
        ),
    )


def _probe_torch() -> dict:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_available": False,
            "cuda_available": None,
            "devices": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": int(props.total_memory),
                }
            )
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "devices": devices,
    }


def _overall_status(checks: list[PreflightCheck]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"
