"""
utils/device.py
===============
Device detection and management for CUDA, Apple Silicon (MPS), and CPU.

The detection priority is:
    1. Forced device override (from config ``device.force``)
    2. NVIDIA CUDA
    3. Apple Silicon MPS
    4. CPU fallback

Usage
-----
>>> from utils.device import get_device
>>> device = get_device()
>>> print(device)
device(type='cuda')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable container describing the detected compute device.

    Attributes
    ----------
    device : torch.device
        The resolved PyTorch device object.
    backend : str
        Human-readable backend name: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    name : str
        Hardware description string (e.g. GPU model name).
    supports_amp : bool
        Whether Automatic Mixed Precision (AMP) is fully supported.
    cuda_version : Optional[str]
        CUDA toolkit version string if the backend is CUDA, else ``None``.
    """

    device: torch.device
    backend: str
    name: str
    supports_amp: bool
    cuda_version: Optional[str] = field(default=None)

    def __str__(self) -> str:  # noqa: D401
        """Return a formatted summary string."""
        amp_str = "✓ AMP supported" if self.supports_amp else "✗ AMP not supported"
        cuda_str = (
            f"  • CUDA version : {self.cuda_version}\n"
            if self.cuda_version
            else ""
        )
        return (
            f"DeviceInfo(\n"
            f"  • backend      : {self.backend}\n"
            f"  • device       : {self.device}\n"
            f"  • hardware     : {self.name}\n"
            f"{cuda_str}"
            f"  • {amp_str}\n"
            f")"
        )


def get_device(force: Optional[str] = None) -> DeviceInfo:
    """Detect and return the best available compute device.

    Parameters
    ----------
    force : Optional[str]
        If provided, override auto-detection and use the given backend.
        Must be one of ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    DeviceInfo
        A frozen dataclass containing the resolved device and metadata.

    Raises
    ------
    ValueError
        If ``force`` is set to an invalid backend string.
    RuntimeError
        If the forced backend is not available on this machine.

    Examples
    --------
    >>> info = get_device()
    >>> model = model.to(info.device)
    """
    if force is not None:
        return _resolve_forced_device(force)

    return _auto_detect_device()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _auto_detect_device() -> DeviceInfo:
    """Run the detection waterfall: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return _build_cuda_info()

    if _mps_is_available():
        return _build_mps_info()

    return _build_cpu_info()


def _resolve_forced_device(force: str) -> DeviceInfo:
    """Validate and resolve a user-forced device string."""
    valid_backends = {"cuda", "mps", "cpu"}
    if force not in valid_backends:
        raise ValueError(
            f"Invalid device override '{force}'. "
            f"Must be one of {valid_backends}."
        )

    if force == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Forced device 'cuda' but CUDA is not available on this machine."
        )

    if force == "mps" and not _mps_is_available():
        raise RuntimeError(
            "Forced device 'mps' but MPS is not available on this machine."
        )

    builder_map = {
        "cuda": _build_cuda_info,
        "mps": _build_mps_info,
        "cpu": _build_cpu_info,
    }
    return builder_map[force]()


def _mps_is_available() -> bool:
    """Return ``True`` if Apple MPS backend is available."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _build_cuda_info() -> DeviceInfo:
    """Build DeviceInfo for CUDA backend."""
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    cuda_ver = torch.version.cuda  # type: ignore[attr-defined]

    info = DeviceInfo(
        device=device,
        backend="cuda",
        name=gpu_name,
        supports_amp=True,
        cuda_version=cuda_ver,
    )
    logger.info("Device detected:\n%s", info)
    return info


def _build_mps_info() -> DeviceInfo:
    """Build DeviceInfo for Apple Silicon MPS backend."""
    device = torch.device("mps")
    # MPS does not expose a hardware name string; use platform info instead
    try:
        import platform
        chip = platform.processor() or "Apple Silicon"
    except Exception:
        chip = "Apple Silicon"

    info = DeviceInfo(
        device=device,
        backend="mps",
        name=chip,
        # MPS supports float16 (AMP) from PyTorch 2.0+
        supports_amp=int(torch.__version__.split(".")[0]) >= 2,
    )
    logger.info("Device detected:\n%s", info)
    return info


def _build_cpu_info() -> DeviceInfo:
    """Build DeviceInfo for CPU fallback."""
    try:
        import platform
        cpu_name = platform.processor() or "CPU"
    except Exception:
        cpu_name = "CPU"

    info = DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        name=cpu_name,
        supports_amp=False,
    )
    logger.warning(
        "No GPU/MPS detected. Running on CPU — training will be slow.\n%s",
        info,
    )
    return info
