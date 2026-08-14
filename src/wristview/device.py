"""Torch device selection for Apple Silicon.

MPS is the target. Two facts shape everything that uses it:

- MPS has no float64. Every tensor must be float32.
- A few index and reduction kernels are missing. `PYTORCH_ENABLE_MPS_FALLBACK`
  routes those to the CPU instead of raising.
"""

from __future__ import annotations

import os

from .logging_setup import get

log = get(__name__)

_RESOLVED: str | None = None


def resolve(preferred: str = "auto", allow_cpu_fallback: bool = True) -> str:
    """Return the torch device string to use for this process."""
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    if allow_cpu_fallback:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch

    if preferred != "auto":
        _RESOLVED = preferred
    elif torch.backends.mps.is_available():
        _RESOLVED = "mps"
    elif torch.cuda.is_available():
        _RESOLVED = "cuda"
    else:
        _RESOLVED = "cpu"

    log.info("torch %s on device %s", torch.__version__, _RESOLVED)
    return _RESOLVED


def dtype_for(device: str):
    """MPS is float32 only. Everything else gets float32 too, for consistency."""
    import torch

    return torch.float32


def sync(device: str) -> None:
    """Block until queued work on the device finishes. Needed for honest timings."""
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def empty_cache(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
