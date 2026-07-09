"""simsopt interop: Optimizable wrappers for coil-fem."""

from .objective import CoilFEMObjective
from .support import CoilSupport, CoilSupportDiscrete, CoilSupportTopBottom

__all__ = [
    "CoilFEMObjective",
    "CoilSupport",
    "CoilSupportDiscrete",
    "CoilSupportTopBottom",
]
