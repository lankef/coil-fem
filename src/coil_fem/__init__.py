"""Differentiable structural FEM analysis for stellarator coils.

Exposes :class:`~coil_fem.CoilFEM`, the main pipeline container.  Simsopt
interop lives in :mod:`coil_fem.simsopt`, magnetic helpers in
:mod:`coil_fem.magnetic`, curve geometry in :mod:`coil_fem.geo`, and the
optional GPU cuDSS backend in :mod:`coil_fem.solvers`.
"""

from .gpu_env import clear_simsopt_cpu_pin
from .coil_fem import CoilFEM

# MUST stay last: CoilFEM → magnetic imports simsopt and applies its global
# jax_platform_name="cpu" pin. Clear after that import, before user code.
clear_simsopt_cpu_pin()

__all__ = ["CoilFEM"]
