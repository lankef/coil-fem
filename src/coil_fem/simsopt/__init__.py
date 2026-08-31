"""Simsopt Optimizable wrappers for coil-fem supports and objectives.

Re-exports :class:`CoilSupport` and its Fixed / Beams / CSR subclasses
(including Sorted variants), plus :class:`CoilFEMObjective` and the
beam / CSR geometric constraints.
"""

from .objectives import (
    BeamSurfaceDistance,
    BeamCurveDistance,
    BeamCurveAngle,
    ClampInboard,
    CoilFEMObjective,
    CRBeamInboard,
    CSRVolume,
    CSRCurveDistance,
    CSRSurfaceDistance,
)
from .coil_support import CoilSupport
from .coil_support_fixed import (
    CoilSupportFixed,
    CoilSupportFixedSorted,
    CoilSupportTopBottom,
)
from .coil_support_beams import CoilSupportBeams, CoilSupportBeamsSorted
from .coil_support_beams_csr import CoilSupportBeamsCSR, CoilSupportBeamsCSRSorted
from .utils import constraint_from_optimizable

__all__ = [
    "BeamSurfaceDistance",
    "BeamCurveDistance",
    "BeamCurveAngle",
    "ClampInboard",
    "CoilFEMObjective",
    "CRBeamInboard",
    "CSRVolume",
    "CSRCurveDistance",
    "CSRSurfaceDistance",
    "CoilSupport",
    "CoilSupportFixed",
    "CoilSupportFixedSorted",
    "CoilSupportTopBottom",
    "CoilSupportBeams",
    "CoilSupportBeamsSorted",
    "CoilSupportBeamsCSR",
    "CoilSupportBeamsCSRSorted",
    "constraint_from_optimizable",
]
