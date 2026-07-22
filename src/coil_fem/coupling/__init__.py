"""Coupling interfaces between coil FEM problems and support structures.

Re-exports :class:`~coil_fem.coupling.supports.Support`,
:class:`~coil_fem.coupling.supports.SupportFixed`,
:class:`~coil_fem.coupling.beam_networks.SupportBeams`, and the driver
functions :func:`~coil_fem.coupling.drivers.solve_staggered` and
:func:`~coil_fem.coupling.drivers.solve_monolithic`.
"""

from .supports import Support, SupportFixed
from .beam_network import SupportBeams
from .drivers import solve_staggered, solve_monolithic

__all__ = [
    'Support',
    'SupportFixed',
    'SupportBeams',
    'solve_staggered',
    'solve_monolithic',
]
