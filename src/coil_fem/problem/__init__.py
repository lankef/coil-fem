"""FEM ``Problem`` subpackage for coil-fem.

Exposes the differentiable linear-elasticity problem and its device-assembly
base class from a single namespace so callers can simply do::

    from coil_fem.problem import LinearElasticity3D, DeviceProblem

``linear_elasticity`` holds :class:`LinearElasticity3D` (Path-C geometry
differentiation, Winkler BCs, thermal eigenstrain) plus material/BC/geometry
and post-processing helpers.  ``device_problem`` holds :class:`DeviceProblem`,
the JAX device-assembly ``Problem`` subclass used by the cuDSS solver backend.
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
