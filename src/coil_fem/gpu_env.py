"""Process-level XLA GPU memory configuration for the cuDSS path.

Must be called *before* any JAX computation (i.e. before the PJRT GPU client
is constructed). Importing this module is safe; only ``configure_gpu_memory``
mutates the environment.
"""
from __future__ import annotations
import os
import warnings

_VARS = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")


def configure_gpu_memory(mem_fraction: float = 0.5, *, force: bool = False) -> dict[str, str]:
    """Leave GPU memory free for cuDSS, which allocates outside XLA's pool.

    cuDSS allocates its factorisation and workspace inside the spineax FFI call,
    using its own allocator. XLA's BFC pool cannot be borrowed from, so cuDSS can 
    end up running with onlyu 25% of the total VRAM (XLA pre-allocates 75%). This 
    function disables pre-allocation and caps XLA. Recommeneded for the cudss 
    backend. Please run BEFORE you import JAX-related libraries.

    Returns the values in effect. Respects pre-existing settings unless ``force``.
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