"""Coupling interfaces between coil FEM problems and support structures.

Re-exports :class:`~coil_fem.coupling.supports.Support`,
:class:`~coil_fem.coupling.beam_network.SupportBeams`,
:class:`~coil_fem.coupling.beam_network_csr.SupportBeamsCSR`, the driver
functions :func:`~coil_fem.coupling.drivers.solve_staggered` and
:func:`~coil_fem.coupling.drivers.solve_monolithic`, and the static bundle
:class:`~coil_fem.coupling.drivers.MonolithicStatic` together with
:func:`~coil_fem.coupling.drivers.build_monolithic_static`.
"""

from .supports import ContinuumMember, Support
from .beam_network import SupportBeams
from .beam_network_csr import SupportBeamsCSR
from .drivers import (
    solve_uncoupled,
    solve_staggered,
    solve_monolithic,
    MonolithicStatic,
    build_monolithic_static,
)

__all__ = [
    'ContinuumMember',
    'Support',
    'SupportBeams',
    'SupportBeamsCSR',
    'solve_uncoupled',
    'solve_staggered',
    'solve_monolithic',
    'MonolithicStatic',
    'build_monolithic_static',
]
