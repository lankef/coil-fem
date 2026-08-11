"""GPU environment helpers for simsopt's JAX CPU pin and XLA memory policy.

1. Simsopt JAX CPU pin

:mod:`simsopt` pins JAX's default device to the CPU process-wide via
``jax_platform_name='cpu'``. ``coil_fem`` clears that pin from
``coil_fem/__init__.py`` after the eager simsopt import in :mod:`coil_fem.magnetic`.
Call :func:`clear_simsopt_cpu_pin` again if you import simsopt *after*
``coil_fem``.

2. XLA GPU pre-allocation (opt-in)

:func:`configure_gpu_memory` sets ``XLA_PYTHON_CLIENT_PREALLOCATE`` / mem
fraction for cuDSS. Call it before any JAX computation; importing this module
does not change those env vars.
"""
from __future__ import annotations

import os
import warnings

import jax

__all__ = ["clear_simsopt_cpu_pin", "configure_gpu_memory"]


# ============================================================================
# Fix 1. Simsopt JAX CPU pin
# ============================================================================


def clear_simsopt_cpu_pin() -> str | None:
    """Clear simsopt's process-wide ``jax_platform_name="cpu"`` pin.

    Restores JAX's normal backend auto-selection: GPU when a CUDA backend is
    registered, CPU otherwise. Unconditional w.r.t. ``JAX_PLATFORMS`` (that
    variable controls which backends are registered; a leftover third-party
    pin must not override it).

    Idempotent. Call again if simsopt is imported after ``coil_fem``.

    Returns
    -------
    str or None
        Pin value before clearing, or ``None`` if unset.
    """
    previous = jax.config.values.get("jax_platform_name") or None

    # A pin cleared after the backend is live cannot move arrays already on a
    # device. Warn rather than fail: the reset is still correct going forward.
    if previous is not None and _backend_is_initialized():
        warnings.warn(
            f"coil_fem cleared a jax_platform_name={previous!r} pin after the "
            "JAX backend was already initialised. Arrays created before this "
            "point may live on the wrong device. Import coil_fem before "
            "running any JAX computation.",
            RuntimeWarning,
            stacklevel=2,
        )

    jax.config.update("jax_platform_name", None)
    return previous


def _backend_is_initialized() -> bool:
    """Best-effort check for a live PJRT backend. Never raises."""
    try:
        from jax._src import xla_bridge
        return bool(getattr(xla_bridge, "_backends", None))
    except Exception:
        return False


# ============================================================================
# Fix 2. XLA GPU pre-allocation (opt-in)
# ============================================================================

_VARS = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")


def configure_gpu_memory(mem_fraction: float = 0.9, *, force: bool = False) -> dict[str, str]:
    """Leave GPU memory free for cuDSS, which allocates outside XLA's pool.

    Disables XLA pre-allocation and caps the BFC pool. Call before importing
    JAX-related libraries. Respects pre-existing env settings unless ``force``.

    Returns
    -------
    dict[str, str]
        Values of the two XLA env vars in effect.
    """
    if "jax" in __import__("sys").modules:
        warnings.warn(
            "configure_gpu_memory() called after `import jax`. This is usually "
            "still fine (the GPU client is constructed lazily), but it has no "
            "effect if any JAX computation has already run.",
            stacklevel=2,
        )
    settings = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
    }
    for k, v in settings.items():
        if force or k not in os.environ:
            os.environ[k] = v
    return {k: os.environ[k] for k in _VARS}
