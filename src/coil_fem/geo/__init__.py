"""Curve geometry, framed curves, and symmetry helpers for stellarator coils."""

from .curve_jax import CurveXYZFourierJAX
from .framed_curve_jax import (
    FramedCurveJAX,
    FramedCurveCentroidJAX,
    FramedCurveRMFJAX,
    make_centroid_frame,
    make_rmf_frame,
    make_framed_curve,
)
from .symmetries import (
    apply_symmetries_to_gammas,
    apply_symmetries_to_gammadashs,
    apply_symmetries_to_currents,
    n_coils_total,
)
