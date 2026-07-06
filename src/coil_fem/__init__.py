"""
coil-fem: Lorentz/body-force densities and JAX-FEM structural analysis for
stellarator coils.

Top-level exports the CoilFEM container plus the small magnetic/force
utilities.  Curve geometry lives in :mod:`coil_fem.geo`, simsopt interop in
:mod:`coil_fem.simsopt`, and the GPU cuDSS backend in
:mod:`coil_fem.backend`.

Equilibrium (VMEC/DESC) is out of scope: callers supply B or equivalent loads.

Modules include meshing (structured beam/coil volume mesh), geo (JAX curves,
framed curves, symmetry expansion), magnetic (B-field helpers), forces
(Lorentz body force), metrics (FEM post-processing objectives),
elasticity / thermal, container (CoilFEM container).
"""

from .container import CoilFEM
from .magnetic import biot_savart, B_self_quadrature
from .forces import lorentz_body_force
