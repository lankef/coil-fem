"""Coupling interfaces between coil FEM problems and support structures.

Re-exports :class:`~coil_fem.coupling.supports.Support`,
:class:`~coil_fem.coupling.beam_networks.SupportBeams`, the factory helpers
:func:`~coil_fem.coupling.supports.make_clamp_fn` and
:func:`~coil_fem.coupling.supports.make_topbottom_fn`, and the driver
functions :func:`~coil_fem.coupling.drivers.solve_staggered` and
:func:`~coil_fem.coupling.drivers.solve_monolithic`.
"""

from .supports import Support, make_clamp_fn, make_topbottom_fn
from .beam_network import SupportBeams
from .drivers import solve_staggered, solve_monolithic

__all__ = [
    'Support',
    'SupportBeams',
    'make_clamp_fn',
    'make_topbottom_fn',
    'solve_staggered',
    'solve_monolithic',
]
