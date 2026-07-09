"""Differentiable structural FEM analysis for stellarator coils.

Exposes :class:`~coil_fem.CoilFEM` (the main pipeline container) and the
magnetic helpers :func:`~coil_fem.biot_savart`,
:func:`~coil_fem.B_self_quadrature`, and
:func:`~coil_fem.lorentz_body_force`.  Curve geometry lives in
:mod:`coil_fem.geo`, simsopt interop in :mod:`coil_fem.simsopt`, and the
optional GPU cuDSS backend in :mod:`coil_fem.solver`.
"""

from .coil_fem import CoilFEM
from .magnetic import biot_savart, B_self_quadrature, lorentz_body_force

__all__ = [
    "CoilFEM",
    "biot_savart",
    "B_self_quadrature",
    "lorentz_body_force",
]
