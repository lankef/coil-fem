"""Differentiable structural FEM analysis for stellarator coils.

Exposes :class:`~coil_fem.CoilFEM`, the main pipeline container.  Simsopt
interop lives in :mod:`coil_fem.simsopt`, magnetic helpers in
:mod:`coil_fem.magnetic`, curve geometry in :mod:`coil_fem.geo`, and the
optional GPU cuDSS backend in :mod:`coil_fem.solvers`.
"""

import os

from .gpu_env import clear_simsopt_cpu_pin
from .coil_fem import CoilFEM

# MUST stay last: CoilFEM → magnetic imports simsopt and applies its global
# jax_platform_name="cpu" pin. Clear after that import, before user code.
# Skip on Read the Docs: Sphinx only needs the import; JAX config APIs vary
# across versions and the pin is irrelevant for the docs build.
if os.environ.get("READTHEDOCS") != "True":
    clear_simsopt_cpu_pin()

__all__ = ["CoilFEM"]
