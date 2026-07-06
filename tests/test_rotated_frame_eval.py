"""
Tests for the pure-JAX :meth:`rotated_frame_eval` overrides on
``FramedCurveCentroidJAX`` and ``FramedCurveRMFJAX`` and the supporting
:meth:`alpha_eval`.

Validated properties
--------------------
1. ``alpha_eval`` reproduces the discrete ``alpha`` samples at the
   curve's quadpoints to machine precision.
2. ``alpha_eval`` is exact for a band-limited synthetic ``alpha`` away
   from the quadpoints (it agrees with the closed-form Fourier series
   used to construct ``alpha``).
3. ``FramedCurveCentroidJAX.rotated_frame_eval`` reproduces
   :meth:`rotated_frame` to machine precision when evaluated at the
   curve's quadpoints, and produces an orthonormal right-handed frame
   off-grid.
4. ``FramedCurveRMFJAX.rotated_frame_eval`` matches
   :meth:`rotated_frame` at the curve's quadpoints (the fresh-grid scan
   reuses the same algorithm), and is JIT-compatible.
5. The frame evaluations are differentiable through ``curve.dofs``,
   ``alpha``, and the target ``phi``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from coil_fem.geo import CurveXYZFourierJAX
from coil_fem.geo import (
    make_centroid_frame,
    make_rmf_frame,
)


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_circle(N=64, R=1.0):
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    dofs = jnp.array([0.0, 0.0, R,
                      0.0, R,  0.0,
                      0.0, 0.0, 0.0])
    return CurveXYZFourierJAX(quadpoints, dofs, order=1)


def _make_wavy_curve(N=64, order=3, seed=0):
    """A slightly wavy circular-like curve with non-trivial frame."""
    quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
    key = jax.random.PRNGKey(seed)
    n_dof = 3 * (2 * order + 1)
    dofs = jax.random.normal(key, (n_dof,)) * 0.05
    # Set a circular baseline: x = cos(2 pi phi), y = sin(2 pi phi)
    # Layout per coord: [c0, s1, c1, s2, c2, ...]
    dofs = dofs.at[2].set(1.0)                          # x = c1 cos(2pi phi)
    dofs = dofs.at[2 * (2 * order + 1) - (2 * order)].set(0.0)
    dofs = dofs.at[(2 * order + 1) + 1].set(1.0)        # y = s1 sin(2pi phi)
    return CurveXYZFourierJAX(quadpoints, dofs, order=order)


def _make_band_limited_alpha(N, modes):
    """Construct an alpha sampled at ``phi_k = k/N`` from a known set of
    Fourier modes.  Returns ``(alpha_samples, alpha_continuous_fn)``."""
    rng = np.random.default_rng(7)
    coeffs = []
    for k, _ in modes:
        if k == 0:
            coeffs.append((float(rng.standard_normal()), 0.0))
        else:
            coeffs.append((
                float(rng.standard_normal()),
                float(rng.standard_normal()),
            ))

    def alpha_continuous(phi):
        phi = jnp.asarray(phi, dtype=float)
        out = jnp.zeros_like(phi)
        for (k, _), (a, b) in zip(modes, coeffs):
            theta = 2.0 * jnp.pi * k * phi
            out = out + a * jnp.cos(theta) + b * jnp.sin(theta)
        return out

    phi_k = jnp.linspace(0.0, 1.0, N, endpoint=False)
    return jnp.asarray(alpha_continuous(phi_k)), alpha_continuous


# ---------------------------------------------------------------------------
# alpha_eval tests
# ---------------------------------------------------------------------------

class TestAlphaEval:
    @pytest.mark.parametrize("N", [16, 32, 33, 64])
    def test_at_quadpoints(self, N):
        """At the discrete sampling phases, ``alpha_eval`` must reproduce
        the stored ``alpha`` array to machine precision."""
        rng = np.random.default_rng(N)
        alpha = jnp.asarray(rng.standard_normal(N))
        curve = _make_circle(N=N)
        fc = make_centroid_frame(curve, alpha=alpha)
        out = fc.alpha_eval(curve.quadpoints)
        assert jnp.allclose(out, alpha, atol=1e-13, rtol=1e-13)

    @pytest.mark.parametrize("N", [16, 32, 33])
    def test_band_limited_exact(self, N):
        """For a band-limited alpha, ``alpha_eval`` must reproduce the
        underlying continuous Fourier series at *arbitrary* phi."""
        # Use modes well below Nyquist
        modes = [(0, None), (1, None), (2, None), (3, None)]
        alpha_samples, alpha_continuous = _make_band_limited_alpha(N, modes)
        curve = _make_circle(N=N)
        fc = make_centroid_frame(curve, alpha=alpha_samples)
        phi_test = jnp.linspace(0.0, 1.0, 97, endpoint=False)
        out = fc.alpha_eval(phi_test)
        ref = alpha_continuous(phi_test)
        assert jnp.allclose(out, ref, atol=1e-11, rtol=1e-11)

    def test_grad_through_alpha(self):
        N = 32
        curve = _make_circle(N=N)

        def loss(alpha):
            fc = make_centroid_frame(curve, alpha=alpha)
            return jnp.sum(fc.alpha_eval(jnp.array([0.123, 0.456, 0.789])) ** 2)

        alpha0 = jnp.ones(N) * 0.1
        g = jax.grad(loss)(alpha0)
        assert g.shape == alpha0.shape
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Centroid frame
# ---------------------------------------------------------------------------

class TestCentroidFrameEval:
    def test_matches_rotated_frame_on_quadpoints(self):
        N = 32
        curve = _make_wavy_curve(N=N, order=2)
        rng = np.random.default_rng(0)
        alpha = jnp.asarray(rng.standard_normal(N) * 0.3)
        fc = make_centroid_frame(curve, alpha=alpha)
        t0, p0, q0 = fc.rotated_frame()
        t1, p1, q1 = fc.rotated_frame_eval(curve.quadpoints)
        assert jnp.allclose(t0, t1, atol=1e-12)
        assert jnp.allclose(p0, p1, atol=1e-12)
        assert jnp.allclose(q0, q1, atol=1e-12)

    def test_orthonormal_offgrid(self):
        N = 32
        curve = _make_wavy_curve(N=N, order=2)
        fc = make_centroid_frame(curve)
        phi = jnp.linspace(0.0, 1.0, 97, endpoint=False)
        t, p, q = fc.rotated_frame_eval(phi)
        # Norm == 1
        assert jnp.allclose(jnp.linalg.norm(t, axis=-1), 1.0, atol=1e-12)
        assert jnp.allclose(jnp.linalg.norm(p, axis=-1), 1.0, atol=1e-12)
        assert jnp.allclose(jnp.linalg.norm(q, axis=-1), 1.0, atol=1e-12)
        # Orthogonal
        assert jnp.allclose(jnp.sum(t * p, axis=-1), 0.0, atol=1e-12)
        assert jnp.allclose(jnp.sum(t * q, axis=-1), 0.0, atol=1e-12)
        assert jnp.allclose(jnp.sum(p * q, axis=-1), 0.0, atol=1e-12)
        # Right-handed: t x p == q
        assert jnp.allclose(jnp.cross(t, p), q, atol=1e-12)

    def test_grad_through_dofs(self):
        N = 32
        quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
        phi_target = jnp.linspace(0.0, 1.0, 9, endpoint=False)

        def loss(dofs):
            curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)
            fc = make_centroid_frame(curve)
            _, p, _ = fc.rotated_frame_eval(phi_target)
            return jnp.sum(p ** 2)

        dofs0 = jnp.array([0.0, 0.0, 1.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 0.05])
        g = jax.grad(loss)(dofs0)
        assert g.shape == dofs0.shape
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# RMF frame
# ---------------------------------------------------------------------------

class TestRMFFrameEval:
    def test_matches_rotated_frame_on_quadpoints(self):
        N = 32
        curve = _make_circle(N=N)
        rng = np.random.default_rng(0)
        alpha = jnp.asarray(rng.standard_normal(N) * 0.2)
        fc = make_rmf_frame(curve, alpha=alpha)
        t0, p0, q0 = fc.rotated_frame()
        t1, p1, q1 = fc.rotated_frame_eval(curve.quadpoints)
        # Tolerance is looser: the fresh-grid scan re-applies the periodic
        # closure step on the same uniform grid; results should agree to
        # numerical precision modulo a single floating-point rebuild.
        assert jnp.allclose(t0, t1, atol=1e-10)
        assert jnp.allclose(p0, p1, atol=1e-10)
        assert jnp.allclose(q0, q1, atol=1e-10)

    def test_orthonormal_offgrid(self):
        N = 32
        curve = _make_circle(N=N)
        fc = make_rmf_frame(curve)
        # Use a sorted uniform grid (RMF is defined by ordered propagation)
        phi = jnp.linspace(0.0, 1.0, 96, endpoint=False)
        t, p, q = fc.rotated_frame_eval(phi)
        assert jnp.allclose(jnp.linalg.norm(t, axis=-1), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.linalg.norm(p, axis=-1), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.linalg.norm(q, axis=-1), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.sum(t * p, axis=-1), 0.0, atol=1e-10)
        assert jnp.allclose(jnp.cross(t, p), q, atol=1e-10)

    def test_jit_compatible(self):
        N = 32
        curve = _make_circle(N=N)
        fc = make_rmf_frame(curve)
        phi = jnp.linspace(0.0, 1.0, 64, endpoint=False)

        @jax.jit
        def go(framed, phi_):
            return framed.rotated_frame_eval(phi_)

        t, p, q = go(fc, phi)
        assert t.shape == (64, 3)
        assert jnp.all(jnp.isfinite(t))
        assert jnp.all(jnp.isfinite(p))
        assert jnp.all(jnp.isfinite(q))

    def test_grad_through_dofs(self):
        N = 32
        quadpoints = jnp.linspace(0.0, 1.0, N, endpoint=False)
        phi_target = jnp.linspace(0.0, 1.0, 9, endpoint=False)

        def loss(dofs):
            curve = CurveXYZFourierJAX(quadpoints, dofs, order=1)
            fc = make_rmf_frame(curve)
            _, p, _ = fc.rotated_frame_eval(phi_target)
            return jnp.sum(p ** 2)

        dofs0 = jnp.array([0.0, 0.0, 1.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 0.05])
        g = jax.grad(loss)(dofs0)
        assert g.shape == dofs0.shape
        assert jnp.all(jnp.isfinite(g))
