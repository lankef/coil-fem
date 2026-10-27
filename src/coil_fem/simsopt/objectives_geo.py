"""LSE curve objectives for simsopt.

:class:`CurveCurveDistanceMinLSE` and :class:`CurveSurfaceDistanceMinLSE`
estimate the smallest sampled distance, and :class:`CurveCurvatureMaxSqLSE`
estimates the largest sampled curvature squared. Each error is at most a
given ``epsilon``. They depend only on simsopt and JAX.
"""

from __future__ import annotations

import math

import numpy as np
from jax import grad, jit
import jax.numpy as jnp
from jax.nn import logsumexp

from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable


def max_lse(x, epsilon):
    r"""Approximate :math:`\max(x)` by a log-sum-exp, to within ``epsilon``.

    For :math:`t > 0` and :math:`n` entries,

    .. math::

        \max x
        \le \frac{1}{t}\operatorname{LSE}(t x)
        \le \max x + \frac{\log n}{t}.

    Taking :math:`t = \max(1, \log n) / \epsilon` makes the added term at most
    ``epsilon`` (and exactly zero error when ``n`` is 1). ``epsilon`` is not
    scaled by ``log n`` directly: that would make the error grow with the
    number of samples.

    Parameters
    ----------
    x : jax.Array
        Values whose maximum is approximated. Every entry is included.
    epsilon : float
        Positive error bound. The result is at most ``epsilon`` above
        :math:`\max(x)`.

    Returns
    -------
    jax.Array, shape ()
        The approximation.
    """
    # x.size is static under jit; math.log keeps the temperature a Python float.
    logn = max(1.0, math.log(x.size))
    epsilon_norm = epsilon / logn
    return epsilon_norm * logsumexp(x / epsilon_norm)


class CurveCurveDistanceMinLSE(Optimizable):
    r"""Soft-minimum distance between sampled points on a set of curves.

    Samples are ``curve.gamma()[::downsample]``, and the pairs are the same
    ones :class:`~simsopt.geo.CurveCurveDistance` considers: every ``(i, j)``
    with ``j < i`` and ``j < num_basecurves``. There is no distance threshold
    and no arclength weight. With :math:`d` those point-to-point distances and
    :math:`t = \max(1, \log n) / \epsilon`,

    .. math::

        J = -\frac{1}{t}\operatorname{LSE}(-t\, d),

    which is a soft minimum,

    .. math::

        \min d - \epsilon \le J \le \min d.

    ``J`` never overestimates the clearance, so a constraint ``J >= d_min``
    keeps every sample pair at least ``d_min`` apart.

    Parameters
    ----------
    curves : sequence of simsopt.geo.Curve
        Curves to compare. DOF parents of this objective.
    epsilon : float
        Error bound of the soft minimum [m]. Must be positive.
    num_basecurves : int, optional
        Only the first this many curves are used as the second member of a
        pair. Defaults to all of them. Useful when the rest are symmetry
        images of the first.
    downsample : int
        Keep every ``downsample``-th quadrature point (default 1).
    """

    def __init__(self, curves, epsilon, num_basecurves=None, downsample=1):
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0; got {epsilon}.")
        self.curves = curves
        self.epsilon = epsilon
        self.downsample = int(downsample)
        n_base = len(curves) if num_basecurves is None else int(num_basecurves)
        self.num_basecurves = n_base
        # Same pairs as simsoptpp.get_pointclouds_closer_than_threshold_within_collection,
        # without a threshold to reject any of them.
        self.pairs = [
            (i, j) for i in range(len(curves)) for j in range(min(i, n_base))
        ]
        self.J_jax = jit(self._J_pure)
        self.dJ_dgammas = jit(grad(self._J_pure))
        super().__init__(depends_on=curves)

    def _J_pure(self, gammas):
        """Soft-min of every sample-to-sample distance of every curve pair."""
        if not self.pairs:
            return jnp.asarray(jnp.inf)
        sampled = [gamma[::self.downsample] for gamma in gammas]
        # ponytail: all pairs, O(n_pairs * n_q^2). No threshold, so the
        # simsoptpp candidate grid cannot drop any. Upgrade: a spatial index.
        dists = jnp.concatenate([
            jnp.linalg.norm(
                sampled[i][:, None, :] - sampled[j][None, :, :], axis=2,
            ).ravel()
            for i, j in self.pairs
        ])
        # min(d) - epsilon <= -max_lse(-d) <= min(d)
        return -max_lse(-dists, self.epsilon)

    def J(self):
        """Soft-minimum curve-curve distance [m]."""
        return self.J_jax([c.gamma() for c in self.curves])

    @derivative_dec
    def dJ(self):
        """Gradient of :meth:`J` with respect to the curve DOFs.

        Returns a :class:`~simsopt._core.derivative.Derivative`.
        ``@derivative_dec`` contracts it to a flat array aligned with
        ``self.x``.
        """
        grads = self.dJ_dgammas([c.gamma() for c in self.curves])
        return sum(
            curve.dgamma_by_dcoeff_vjp(np.asarray(g))
            for curve, g in zip(self.curves, grads)
        )

    return_fn_map = {'J': J, 'dJ': dJ}


class CurveSurfaceDistanceMinLSE(Optimizable):
    r"""Soft-minimum distance between sampled curve points and a surface.

    Curve samples are ``curve.gamma()`` and surface samples are
    ``surface.gamma()`` flattened to points, the same quadrature
    :class:`~simsopt.geo.CurveSurfaceDistance` uses. There is no threshold
    and no arclength or area weight. With :math:`d` those point-to-point
    distances and :math:`t = \max(1, \log n) / \epsilon`,

    .. math::

        J = -\frac{1}{t}\operatorname{LSE}(-t\, d),

    so :math:`\min d - \epsilon \le J \le \min d`. ``J`` never overestimates
    the clearance.

    Parameters
    ----------
    curves : sequence of simsopt.geo.Curve
        Curves to compare. DOF parents of this objective.
    surface : simsopt.geo.Surface
        Target surface. Not a DOF parent, so it contributes no gradient,
        matching :class:`~simsopt.geo.CurveSurfaceDistance`.
    epsilon : float
        Error bound of the soft minimum [m]. Must be positive.
    """

    def __init__(self, curves, surface, epsilon):
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0; got {epsilon}.")
        self.curves = curves
        self.surface = surface
        self.epsilon = epsilon
        self.J_jax = jit(self._J_pure)
        self.dJ_dgammac = jit(grad(self._J_pure, argnums=0))
        super().__init__(depends_on=curves)

    def _J_pure(self, gammacs, gammas):
        """Soft-min of every curve-sample to surface-point distance."""
        # ponytail: all pairs, O(n_curves * n_q * n_surf). No threshold, so
        # the simsoptpp candidate grid cannot drop any. Upgrade: a spatial index.
        dists = jnp.concatenate([
            jnp.linalg.norm(gc[:, None, :] - gammas[None, :, :], axis=2).ravel()
            for gc in gammacs
        ])
        # min(d) - epsilon <= -max_lse(-d) <= min(d)
        return -max_lse(-dists, self.epsilon)

    def _surface_points(self):
        """Surface quadrature points, shape ``(M, 3)``."""
        return self.surface.gamma().reshape((-1, 3))

    def J(self):
        """Soft-minimum curve-surface distance [m]."""
        return self.J_jax(
            [c.gamma() for c in self.curves], self._surface_points(),
        )

    @derivative_dec
    def dJ(self):
        """Gradient of :meth:`J` with respect to the curve DOFs.

        The surface is not a DOF parent, so its geometry gets no gradient.
        ``@derivative_dec`` contracts the result to a flat array aligned
        with ``self.x``.
        """
        grads = self.dJ_dgammac(
            [c.gamma() for c in self.curves], self._surface_points(),
        )
        return sum(
            curve.dgamma_by_dcoeff_vjp(np.asarray(g))
            for curve, g in zip(self.curves, grads)
        )

    return_fn_map = {'J': J, 'dJ': dJ}


class CurveCurvatureMaxSqLSE(Optimizable):
    r"""Soft-maximum of curvature squared over a set of curves.

    Samples are ``curve.kappa()`` at each curve's quadrature points, the same
    points :class:`~simsopt.geo.LpCurveCurvature` uses. There is no arclength
    weight and no threshold. With :math:`\kappa^2` those samples over every
    curve and :math:`t = \max(1, \log n) / \epsilon`,

    .. math::

        J = \frac{1}{t}\operatorname{LSE}(t\, \kappa^2),

    so

    .. math::

        \max \kappa^2 \le J \le \max \kappa^2 + \epsilon.

    ``J`` never underestimates the maximum, so a constraint ``J <= \kappa_0^2``
    keeps every sample at or below curvature :math:`\kappa_0`.

    Parameters
    ----------
    curves : sequence of simsopt.geo.Curve
        Curves in the coil set. DOF parents of this objective.
    epsilon : float
        Error bound on :math:`\kappa^2` [1/m²]. Must be positive.
    """

    def __init__(self, curves, epsilon):
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0; got {epsilon}.")
        self.curves = curves
        self.epsilon = epsilon
        self.J_jax = jit(self._J_pure)
        self.dJ_dkappas = jit(grad(self._J_pure))
        super().__init__(depends_on=curves)

    def _J_pure(self, kappas):
        """Soft-max of curvature squared at every quadrature point."""
        sq = jnp.concatenate([kappa ** 2 for kappa in kappas])
        # max(κ^2) <= max_lse(κ^2) <= max(κ^2) + epsilon
        return max_lse(sq, self.epsilon)

    def J(self):
        """Soft-maximum of curvature squared [1/m²]."""
        return self.J_jax([c.kappa() for c in self.curves])

    @derivative_dec
    def dJ(self):
        """Gradient of :meth:`J` with respect to the curve DOFs.

        ``kappa`` already chains through the tangent and its derivative, so
        the only vector-Jacobian product is
        :meth:`~simsopt.geo.Curve.dkappa_by_dcoeff_vjp`.
        ``@derivative_dec`` contracts the result to a flat array aligned
        with ``self.x``.
        """
        grads = self.dJ_dkappas([c.kappa() for c in self.curves])
        return sum(
            curve.dkappa_by_dcoeff_vjp(np.asarray(g))
            for curve, g in zip(self.curves, grads)
        )

    return_fn_map = {'J': J, 'dJ': dJ}
