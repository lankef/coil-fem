"""Coupling interfaces between coil FEM problems and support structures.

Re-exports :class:`~coil_fem.coupling.supports.Support` and
:class:`~coil_fem.coupling.supports.SupportFixed` for convenient import from
``coil_fem.coupling``.
"""

from .supports import Support, SupportFixed

__all__ = ['Support', 'SupportFixed']
