"""Physical systems implemented as ``jax_fem.Problem`` classes.

Exposes :class:`LinearElasticity3D` (differentiable linear-elasticity with
Winkler BCs and thermal eigenstrain), :class:`DeviceProblem` (JAX
device-assembly base class for the cuDSS solver backend), and the stub
:class:`HeatConduction3D` from a single namespace::

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
from .heat_conduction import HeatConduction3D

__all__ = [
    "DeviceProblem",
    "LinearElasticity3D",
    "lame_parameters",
    "itc_strain",
    "dirichlet_bc",
    "recompute_fe_geometry",
    "HeatConduction3D",
]
