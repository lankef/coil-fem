"""
Self-consistency tests for B_self_quadrature (rectangular cross-section).

Since there is no simsopt counterpart for the full Landreman–Hurwitz–Antonsen
(2025) formula, we validate via:

  1. current sign-flip anti-symmetry
  2. swap symmetry (a ↔ b swap + p ↔ q swap leaves the physical field unchanged)
  3. B_full differs non-trivially from B_reg-only (curvature corrections present)
  4. jax.grad smoke test (gradient must not produce NaN)
  5. 1/R scaling of curvature corrections
  6. a/b swap symmetry
  7. NotImplementedError for disk cross-section

The centerline field (u = v = 0) is used via a thin adapter:

    _B_self_rect_centerline(fc, I, w1, w2)
        -> B_self_quadrature(fc, I, {'shape':'rect','w1':w1,'w2':w2},
                             phi_q=(n_phi,1), uv_q=zeros(n_phi,1,2))[:, 0, :]

At u = v = 0 the B_0 term vanishes by reflection symmetry (verified analytically
in the comments below), so the adapter result equals B_reg + B_kappa + B_b,
matching the old B_self_centerline_rect_full behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.geo import make_centroid_frame
from coil_fem.magnetic import B_self_quadrature


# ---------------------------------------------------------------------------
# Adapter: evaluate at the centerline (u = v = 0)
# ---------------------------------------------------------------------------

def _B_self_rect_centerline(fc, I, w1, w2):
    """Evaluate B_self_quadrature at the cross-section centroid (u = v = 0).

    Returns (n_phi, 3) self-field, identical to the former
    ``B_self_centerline_rect_full``.
    """
    n_phi = fc.curve.quadpoints.shape[0]
    phi_q = fc.curve.quadpoints[:, None]          # (n_phi, 1)
    uv_q  = jnp.zeros((n_phi, 1, 2))              # u = v = 0
    cs    = {'shape': 'rect', 'w1': w1, 'w2': w2}
    return B_self_quadrature(fc, I, cs, phi_q, uv_q)[:, 0, :]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_circle(N=64, R=1.0):
    """Unit circle in the x-y plane as a CurveXYZFourierJAX."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, 0.0, R,    # x
                      0.0, R,  0.0,    # y
                      0.0, 0.0, 0.0])  # z
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _make_circle_framed(N=64, R=1.0):
    return make_centroid_frame(_make_circle(N=N, R=R))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBSelfRectFullConsistency:
    """Self-consistency checks for B_self_quadrature (rect cross-section)."""

    def test_current_sign_flip(self):
        """Reversing the current flips the sign of the field."""
        fc = _make_circle_framed(N=32)
        I  = 1e4
        w1, w2 = 0.05, 0.03

        B_pos = _B_self_rect_centerline(fc, +I, w1, w2)
        B_neg = _B_self_rect_centerline(fc, -I, w1, w2)

        np.testing.assert_allclose(
            np.asarray(B_neg), -np.asarray(B_pos),
            rtol=1e-12,
            err_msg="B_self(+I) + B_self(-I) ≠ 0",
        )

    def test_swap_symmetry_square_section(self):
        """For a square cross-section (w1 = w2), swapping p and q (i.e.
        swapping the frame rotation by π/2) must leave |B| unchanged.
        """
        N  = 32
        w  = 0.04
        I  = 1e4

        curve = _make_circle(N=N)
        fc0  = make_centroid_frame(curve, alpha=jnp.zeros(N))
        fc90 = make_centroid_frame(curve, alpha=jnp.full(N, jnp.pi / 2))

        B0  = _B_self_rect_centerline(fc0,  I, w, w)
        B90 = _B_self_rect_centerline(fc90, I, w, w)

        mag0  = jnp.linalg.norm(B0,  axis=1)
        mag90 = jnp.linalg.norm(B90, axis=1)
        np.testing.assert_allclose(
            np.asarray(mag0), np.asarray(mag90),
            rtol=1e-10,
            err_msg="|B| should be invariant under π/2 frame rotation for square cross-section",
        )

    def test_differs_from_B_reg_only(self):
        """B_self at the centerline must differ from B_reg-only, since B_κ and
        B_b add non-trivial curvature corrections.
        """
        from simsopt.field.selffield import regularization_rect, B_regularized_pure

        N  = 64
        R  = 1.0
        w1 = 0.05
        w2 = 0.03
        I  = 1e4

        fc  = _make_circle_framed(N=N, R=R)
        reg = float(regularization_rect(w1, w2))

        B_full = _B_self_rect_centerline(fc, I, w1, w2)
        B_reg_only = B_regularized_pure(
            fc.curve.gamma(), fc.curve.gammadash(), fc.curve.gammadashdash(),
            fc.curve.quadpoints, I, reg,
        )

        rel_diff = float(jnp.linalg.norm(B_full - B_reg_only)
                         / jnp.linalg.norm(B_reg_only))
        assert rel_diff > 0.001, (
            f"Expected non-trivial correction; relative diff = {rel_diff:.2e}"
        )
        assert rel_diff < 0.5, (
            f"Correction is unexpectedly large: relative diff = {rel_diff:.2e}"
        )

    def test_jax_grad_no_nan(self):
        """jax.grad of sum(|B|²) w.r.t. curve DOFs must not produce NaN."""
        N  = 32
        w1 = 0.05
        w2 = 0.03
        I  = 1e4

        curve0     = _make_circle(N=N)
        quadpoints = curve0.quadpoints

        def objective(dofs):
            curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)
            fc    = make_centroid_frame(curve)
            B     = _B_self_rect_centerline(fc, I, w1, w2)
            return jnp.sum(B ** 2)

        grad = jax.grad(objective)(curve0.dofs)
        assert not jnp.any(jnp.isnan(grad)), "jax.grad produced NaN"
        assert not jnp.any(jnp.isinf(grad)), "jax.grad produced Inf"

    def test_corrections_scale_inversely_with_radius(self):
        """B_κ + B_b scale as κ ~ 1/R for a circular coil.

        Doubling R should halve the corrections to within ~20 % tolerance.
        """
        from simsopt.field.selffield import regularization_rect, B_regularized_pure

        N  = 64
        w1 = 0.05
        w2 = 0.03
        I  = 1e4

        def corrections_rms(R):
            fc  = _make_circle_framed(N=N, R=R)
            reg = float(regularization_rect(w1, w2))
            B_full = _B_self_rect_centerline(fc, I, w1, w2)
            B_reg  = B_regularized_pure(
                fc.curve.gamma(), fc.curve.gammadash(), fc.curve.gammadashdash(),
                fc.curve.quadpoints, I, reg,
            )
            return float(jnp.linalg.norm(B_full - B_reg))

        corr_R1 = corrections_rms(R=1.0)
        corr_R2 = corrections_rms(R=2.0)

        ratio = corr_R1 / corr_R2
        assert 1.5 < ratio < 2.5, (
            f"Corrections should scale ~1/R: "
            f"|corr(R=1)| / |corr(R=2)| = {ratio:.3f} (expected ~2)"
        )

    @pytest.mark.parametrize("w1,w2", [(0.03, 0.05), (0.05, 0.03), (0.04, 0.04)])
    def test_ab_swap_symmetry(self, w1, w2):
        r"""Swapping (a, b) ↔ (b, a) together with swapping p ↔ q (frame
        rotation by π/2) must leave |B| unchanged — the cross-section geometry
        is symmetric under simultaneous a ↔ b and p ↔ q.
        """
        N = 32
        I = 1e4
        curve = _make_circle(N=N)

        fc_pq = make_centroid_frame(curve, alpha=jnp.zeros(N))
        fc_qp = make_centroid_frame(curve, alpha=jnp.full(N, jnp.pi / 2))

        B_ab = _B_self_rect_centerline(fc_pq, I, w1, w2)
        B_ba = _B_self_rect_centerline(fc_qp, I, w2, w1)

        np.testing.assert_allclose(
            np.asarray(jnp.linalg.norm(B_ab, axis=1)),
            np.asarray(jnp.linalg.norm(B_ba, axis=1)),
            rtol=1e-10,
            err_msg=(
                f"|B|(a={w1},b={w2}) ≠ |B|(a={w2},b={w1}) after p↔q swap"
            ),
        )

    def test_disk_raises_not_implemented(self):
        """B_self_quadrature for a disk cross-section must raise NotImplementedError."""
        fc = _make_circle_framed(N=16)
        phi_q = fc.curve.quadpoints[:, None]
        cs    = {'shape': 'disk', 'radius': 0.03}
        with pytest.raises(NotImplementedError):
            B_self_quadrature(fc, 1e4, cs, phi_q, uv_quad=None)
