"""GPU environment fixes

This module contains 2 fixes:

1. Simsopt JAX settings fix

:mod:`simsopt` pins JAX's default device to the CPU process-wide. ``coil_fem``
is a GPU-first package and hard-depends on simsopt, so that pin has to be
cleared or every GPU install silently runs on the host.

This module must be imported before any JAX *computation* happens.  It is
imported at the top of ``coil_fem/__init__.py`` for that reason; nothing else
in the package should depend on import ordering to get correct behaviour.

Deliberately *not* handled here: GPU memory policy
(``XLA_PYTHON_CLIENT_PREALLOCATE`` and friends).  Repairing another library's
global mutation is a legitimate import-time side effect, because leaving it
broken is not an option and the correct value is unambiguous.  Choosing a
memory policy is not -- there is a defensible default either way, so it belongs
to the deployment.  See :mod:`coil_fem.gpu_env` for the opt-in helper.

2. XLA GPU pre-allocation fix

Must be called *before* any JAX computation (i.e. before the PJRT GPU client
is constructed). Importing this module is safe; only ``configure_gpu_memory``
mutates the environment.
"""
from __future__ import annotations
import os
import warnings


import jax

__all__ = ["clear_simsopt_cpu_pin"]


# ----- Fix 1. Simsopt JAX settings -----


def clear_simsopt_cpu_pin(*, force_simsopt_import: bool = True) -> str | None:
    """Clear simsopt's process-wide ``jax_platform_name="cpu"`` pin.

    Restores JAX's normal backend auto-selection: GPU when a CUDA backend is
    registered, CPU otherwise.

    This is unconditional with respect to ``JAX_PLATFORMS``.  That variable is
    the more specific mechanism -- it controls which backends are *registered*
    and their priority order -- so a leftover third-party ``jax_platform_name``
    pin must never override it:

    ==========================  ==================================================
    ``JAX_PLATFORMS``           Behaviour if the pin were left in place
    ==========================  ==================================================
    unset                       Runs on CPU even where a GPU is available.
    ``cpu``                     Correct, but ``JAX_PLATFORMS`` alone already
                                gives this -- the pin is redundant.
    ``cuda``                    ``RuntimeError: Unknown backend cpu`` -- the pin
                                requests a backend that was never registered.
    ``cuda,cpu``                *Silently* runs on CPU, defeating the stated
                                priority order.  The worst outcome of the four.
    ==========================  ==================================================

    Idempotent.  Call it again if you import simsopt *after* ``coil_fem``
    (simsopt re-applies the pin on its own first import, and this module cannot
    intercept that).

    Parameters
    ----------
    force_simsopt_import : bool
        Import ``simsopt.geo`` first so the pin is guaranteed to have been
        applied before it is cleared.  Clearing a pin that has not been set yet
        is a no-op that simsopt would immediately undo.  Set ``False`` only if
        simsopt is known to be imported already.

    Returns
    -------
    str or None
        The pin value that was in effect before clearing, or ``None`` if no pin
        was set.  Useful in tests.
    """
    if force_simsopt_import:
        try:
            import simsopt.geo  # noqa: F401  -- imported for its side effect
        except ImportError:
            # simsopt is a hard dependency; let the real import error surface
            # from the module that actually needs it rather than from here.
            pass

    previous = jax.config.jax_platform_name

    # A pin set *after* the backend is live cannot be undone retroactively for
    # arrays already committed to a device. Warn rather than fail: the reset is
    # still correct for everything allocated from here on.
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


# Applied on import: see the module docstring.
clear_simsopt_cpu_pin()


# ----- Fix 2. Simsopt JAX XLA GPU pre-allocation -----


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