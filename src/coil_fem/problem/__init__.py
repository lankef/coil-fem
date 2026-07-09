"""FEM ``Problem`` subpackage for coil-fem.

Exposes :class:`LinearElasticity3D` (differentiable linear-elasticity with
Winkler BCs and thermal eigenstrain) and :class:`DeviceProblem` (JAX
device-assembly base class for the cuDSS solver backend) from a single
namespace::

    from coil_fem.problem import LinearElasticity3D, DeviceProblem
"""

from .device_problem import DeviceProblem
from .linear_elasticity import (
    LinearElasticity3D,
    lame_parameters,
    itc_strain,
    dirichlet_bc,
    recompute_fe_geometry,
)

__all__ = [
    "DeviceProblem",
    "LinearElasticity3D",
    "lame_parameters",
    "itc_strain",
    "dirichlet_bc",
    "recompute_fe_geometry",
]
