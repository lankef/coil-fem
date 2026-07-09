"""
coil-fem: Lorentz/body-force densities and JAX-FEM structural analysis for
stellarator coils.

Top-level exports the CoilFEM container plus the small magnetic/force
utilities.  Curve geometry lives in :mod:`coil_fem.geo`, simsopt interop in
:mod:`coil_fem.simsopt`, and the GPU cuDSS backend in
:mod:`coil_fem.solver`.

Equilibrium (VMEC/DESC) is out of scope: callers supply B or equivalent loads.

Modules include meshing (structured beam/coil volume mesh), geo (JAX curves,
framed curves, symmetry expansion), magnetic (B-field helpers + Lorentz body
force), metrics (FEM post-processing objectives), problem (LinearElasticity3D /
DeviceProblem), and coil_fem (CoilFEM container).
"""

from .coil_fem import CoilFEM
from .magnetic import biot_savart, B_self_quadrature, lorentz_body_force

__all__ = [
    "CoilFEM",
    "biot_savart",
    "B_self_quadrature",
    "lorentz_body_force",
]
