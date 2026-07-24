"""Coupling interfaces between coil FEM problems and support structures.

Re-exports :class:`~coil_fem.coupling.supports.Support`,
:class:`~coil_fem.coupling.beam_networks.SupportBeams`, the driver
functions :func:`~coil_fem.coupling.drivers.solve_staggered` and
:func:`~coil_fem.coupling.drivers.solve_monolithic`, and the static bundle
:class:`~coil_fem.coupling.drivers.MonolithicStatic` together with its
factory :func:`~coil_fem.coupling.drivers.make_merged_solve`.
"""

from .supports import Support
from .beam_network import SupportBeams
from .drivers import solve_staggered, solve_monolithic, MonolithicStatic, make_merged_solve

__all__ = [
    'Support',
    'SupportBeams',
    'solve_staggered',
    'solve_monolithic',
    'MonolithicStatic',
    'make_merged_solve',
]
