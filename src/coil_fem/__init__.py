"""Differentiable structural FEM analysis for stellarator coils.

Exposes :class:`~coil_fem.CoilFEM`, the main pipeline container.  Simsopt
interop lives in :mod:`coil_fem.simsopt`, magnetic helpers in
:mod:`coil_fem.magnetic`, curve geometry in :mod:`coil_fem.geo`, and the
optional GPU cuDSS backend in :mod:`coil_fem.solvers`.
"""

# Must precede every other import: clears simsopt's process-wide JAX CPU pin
# before any JAX computation can happen. See coil_fem._jax_compat.
from . import gpu_env  # noqa: F401

from .coil_fem import CoilFEM

# Re-clear: the import above pulls in simsopt submodules that _jax_compat's own
# `import simsopt.geo` may not have covered, any of which could have re-applied
# the pin. Cheap, idempotent.
gpu_env.clear_simsopt_cpu_pin(force_simsopt_import=False)

__all__ = ["CoilFEM"]