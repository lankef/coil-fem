"""
Pure JAX framed-curve wrappers for :class:`~coil_fem.geo.CurveJAX`.

Provides two reference frames, both fully differentiable through JAX:

* :class:`FramedCurveCentroidJAX` — centroid frame (Singh et al. 2020)
* :class:`FramedCurveRMFJAX` — rotation-minimizing / Bishop frame
  (Wang et al. 2008, double-reflection algorithm)

All operations are pure JAX and require **no** simsopt installation.

When simsopt is available, :func:`make_centroid_frame` and
:func:`make_rmf_frame` also accept simsopt ``CurveXYZFourier`` /
``CurveRZFourier`` objects (converted via :func:`curve_jax_from_simsopt`).

Frame curvatures
----------------
Given a rotated frame :math:`(\\mathbf{t}, \\mathbf{p}, \\mathbf{q})`, the
three curvature scalars :math:`\\kappa_1, \\kappa_2, \\kappa_3` satisfy

.. math::

    \\frac{d}{d\\phi}
    \\begin{pmatrix} \\mathbf{t} \\\\ \\mathbf{p} \\\\ \\mathbf{q} \\end{pmatrix}
    = \\left|\\frac{d\\mathbf{r}_c}{d\\phi}\\right|
    \\begin{pmatrix} 0 & \\kappa_1 & \\kappa_2 \\\\
                    -\\kappa_1 & 0 & \\kappa_3 \\\\
                    -\\kappa_2 & -\\kappa_3 & 0 \\end{pmatrix}
    \\begin{pmatrix} \\mathbf{t} \\\\ \\mathbf{p} \\\\ \\mathbf{q} \\end{pmatrix},

so :math:`\\kappa_1 = (d\\mathbf{t}/dl)\\cdot\\mathbf{p}`,
:math:`\\kappa_2 = (d\\mathbf{t}/dl)\\cdot\\mathbf{q}` (= simsopt
``frame_binormal_curvature``), and
:math:`\\kappa_3 = (d\\mathbf{p}/dl)\\cdot\\mathbf{q}` (= simsopt
``frame_torsion``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from .symmetries import rodrigues as _rodrigues_unit


# ============================================================================
# Pure-JAX RMF helpers  (Wang et al. 2008 double-reflection algorithm)
# ============================================================================

# Squared-eps offset (~1e-30 → forward bias ~1e-15, well below float64
# precision).  Used to make ``||x||`` differentiable at ``x = 0``: the
# gradient ``x / ||x||`` becomes ``x / sqrt(||x||² + ε²)``, which is
# ``0`` at ``x = 0`` instead of NaN.  Specifically needed for the RMF
# periodic-closure step on planar / highly symmetric curves where the
# frame closes exactly and ``cross(n_final, n0)`` is zero.
_SAFE_NORM_EPS2 = 1.0e-30


def _safe_norm(x, axis=-1):
    """Differentiable vector norm with a well-defined gradient at zero.

    Returns ``sqrt(||x||² + ε²)``.  Forward value matches ``||x||`` to
    ~1e-15; backward gradient is ``x / sqrt(||x||² + ε²)``, which is
    ``0`` (not NaN) at ``x = 0``.
    """
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + _SAFE_NORM_EPS2)


def _angle_axis_rotation_matrix(axis, angle):
    """Rodrigues rotation matrix; normalises axis before delegating."""
    return _rodrigues_unit(axis / jnp.linalg.norm(axis), angle)


_angle_axis_rotation_matrix_vmap = jax.jit(jax.vmap(
    _angle_axis_rotation_matrix, in_axes=(0, 0)
))


def _rmf_normals_pure_jax(gamma, gammadash):
    """Compute RMF normal vectors for a closed curve in pure JAX.

    Uses the double-reflection algorithm (Wang et al. 2008) with a
    uniform periodic angular correction.  The seed is
    ``cross(t[0], gamma[0] − centroid)``.

    The residual periodicity gap scales as O(1/N) — use ≥ 64 quadrature
    points for coil mesh sweeps.

    Parameters
    ----------
    gamma : jnp.ndarray, shape (N, 3)
    gammadash : jnp.ndarray, shape (N, 3)  (not normalised)

    Returns
    -------
    normals : jnp.ndarray, shape (N, 3)  — unit RMF normals
    """
    N = gamma.shape[0]
    t = gammadash / jnp.linalg.norm(gammadash, axis=1, keepdims=True)

    # Seed vector: cross(t[0], gamma[0] - centroid)
    centroid = jnp.mean(gamma, axis=0)
    n0_raw = jnp.cross(t[0], gamma[0] - centroid)
    n0 = n0_raw / jnp.linalg.norm(n0_raw)

    def rmf_step(n_prev, x):
        pos_i, pos_ip1, t_i, t_ip1 = x
        v1 = pos_ip1 - pos_i
        c1 = jnp.dot(v1, v1)
        c1 = jnp.where(c1 < 1e-30, 1e-30, c1)
        r_L = n_prev - (2. / c1) * jnp.dot(n_prev, v1) * v1
        t_L = t_i    - (2. / c1) * jnp.dot(t_i,   v1) * v1
        v2 = t_ip1 - t_L
        c2 = jnp.dot(v2, v2)
        c2 = jnp.where(c2 < 1e-30, 1e-30, c2)
        n_i = r_L - (2. / c2) * jnp.dot(r_L, v2) * v2
        return n_i, n_i

    n_final, normals_rest = jax.lax.scan(
        rmf_step, n0,
        (gamma[:-1], gamma[1:], t[:-1], t[1:]),
    )
    normals = jnp.concatenate([n0[None, :], normals_rest], axis=0)  # (N, 3)

    # Periodic correction: determine signed angle (around t[-1]) from
    # n_final towards n0, then distribute uniformly over all points.
    # Use atan2 instead of arccos for a numerically stable gradient when
    # n_final ≈ n0 (dot product ≈ 1 → arccos gradient explodes).
    #
    # Use ``_safe_norm`` rather than ``jnp.linalg.norm``: planar /
    # symmetric coils close the RMF exactly so ``cross(n_final, n0)``
    # vanishes, and ``d(||x||)/dx = x/||x||`` is 0/0 there, producing
    # NaN VJPs.  ``_safe_norm`` injects a 1e-30 squared offset that
    # leaves the forward value unchanged to float64 precision while
    # giving a well-defined zero gradient at ``x = 0``.
    cross_nf_n0 = jnp.cross(n_final, n0)
    sin_raw = _safe_norm(cross_nf_n0)
    cos_raw = jnp.dot(n_final, n0)
    raw_angle = jnp.arctan2(sin_raw, cos_raw)
    R_test = _angle_axis_rotation_matrix(t[-1], raw_angle)
    test_n = R_test @ n_final
    cross_test = jnp.cross(test_n, n0)
    angle_corr = jnp.where(
        jnp.arctan2(_safe_norm(cross_test), jnp.dot(test_n, n0)) < raw_angle,
        raw_angle,
        -raw_angle,
    )

    correction_angles = jnp.linspace(0., angle_corr, N)
    R_corr = _angle_axis_rotation_matrix_vmap(t, correction_angles)
    return jnp.einsum("nij,nj->ni", R_corr, normals)


# ============================================================================
# Frame evaluation helpers
# ============================================================================

def _trig_interp(values, phi):
    """Band-limited periodic interpolation of a uniformly sampled scalar field.

    ``values`` are samples of a 1-periodic scalar at ``phi_k = k/N``.  The
    band-limited Fourier series is reconstructed and evaluated at arbitrary
    *phi*, reproducing ``values`` exactly at the sample points.  Smooth
    (C-infinity) everywhere, so ``jax.grad`` through *phi* is exact.

    Parameters
    ----------
    values : jax.Array, shape (N,)
        Samples on the uniform grid ``phi_k = k/N``.
    phi : array-like
        Target parameter values, arbitrary shape.

    Returns
    -------
    jax.Array
        Shape ``phi.shape``.
    """
    phi_arr = jnp.asarray(phi, dtype=float)
    N = values.shape[0]
    hat = jnp.fft.rfft(values)                        # (K,), K = N//2 + 1
    K = hat.shape[0]
    k = jnp.arange(K, dtype=float)                    # (K,)

    angle = 2.0 * jnp.pi * k * phi_arr[..., None]     # (..., K)
    real = hat.real * jnp.cos(angle) - hat.imag * jnp.sin(angle)

    # Reconstruction weights: 1 for DC; 2 for general k; 1 for Nyquist (only
    # present when N is even).  Matches the band-limited periodic interpolant
    # that ``irfft`` produces at the sampling phases.
    if N % 2 == 0:
        weights = jnp.where((k == 0) | (k == K - 1), 1.0, 2.0)
    else:
        weights = jnp.where(k == 0, 1.0, 2.0)

    return jnp.sum(weights * real, axis=-1) / N


def _centroid_reference(gamma, gammadash, centroid):
    """Closed-form centroid reference frame ``(t, p, q)`` at arbitrary points.

    Shared by the centroid frame itself and by the RMF twist representation,
    which stores the RMF as a rotation angle relative to this frame.
    """
    t = gammadash / jnp.linalg.norm(gammadash, axis=-1, keepdims=True)
    delta = gamma - centroid
    p = delta - jnp.sum(delta * t, axis=-1, keepdims=True) * t
    p = p / _safe_norm(p, axis=-1)[..., None]
    q = jnp.cross(t, p)
    return t, p, q


def _rmf_twist_pure(gamma, gammadash):
    """RMF twist angle relative to the centroid frame, per quadpoint.

    Returns ``theta`` such that the (pre-``alpha``) RMF normal satisfies
    ``p_rmf = cos(theta) p_c - sin(theta) q_c`` in the centroid reference
    frame.  Reducing the frame to a single scalar per quadpoint is what makes
    smooth off-grid evaluation possible: interpolating this angle keeps
    ``(t, p, q)`` exactly orthonormal by construction, whereas interpolating
    the vectors componentwise does not.

    Parameters
    ----------
    gamma, gammadash : jax.Array, shape (N, 3)

    Returns
    -------
    jax.Array, shape (N,)
    """
    t = gammadash / jnp.linalg.norm(gammadash, axis=1, keepdims=True)
    p_rmf = _rmf_normals_pure_jax(gamma, gammadash)
    # Re-orthogonalise exactly as _rotated_rmf_frame_pure does, so the stored
    # twist reproduces that function's frame.
    p_rmf = p_rmf - jnp.sum(p_rmf * t, axis=1, keepdims=True) * t
    p_rmf = p_rmf / jnp.linalg.norm(p_rmf, axis=1, keepdims=True)

    _, p_c, q_c = _centroid_reference(gamma, gammadash, jnp.mean(gamma, axis=0))
    return jnp.arctan2(
        -jnp.sum(p_rmf * q_c, axis=1), jnp.sum(p_rmf * p_c, axis=1)
    )


def _unwrap_periodic(theta):
    """Split a sampled angle field into a linear winding ramp and a periodic part.

    ``theta`` is sampled at ``phi_k = k/N`` and is continuous only modulo 2pi.
    The RMF's periodic closure correction (:func:`_rmf_normals_pure_jax`) leaves
    an O(1/N) residual, so ``theta`` is not exactly periodic and interpolating
    it directly would ring at the ``phi = 0`` seam.  Unwrapping over one full
    period (including the wrap step back to index 0) and subtracting the linear
    ramp leaves an exactly periodic remainder that is safe to interpolate.

    Returns
    -------
    periodic : jax.Array, shape (N,)
        Exactly periodic remainder, for :func:`_trig_interp`.
    total : jax.Array, scalar
        Net winding over one full period; add ``total * phi`` analytically.
    """
    N = theta.shape[0]
    # Include the wrap step so the winding is measured over the full period.
    ext = jnp.unwrap(jnp.concatenate([theta, theta[:1]]))
    total = ext[-1] - ext[0]
    ramp = jnp.arange(N, dtype=float) * (total / N)
    return ext[:-1] - ramp, total


@jax.jit
def _twist_jitted(gamma, gammadash):
    """JIT-cached RMF twist.

    The twist runs the ``lax.scan`` double-reflection sweep.  Evaluated eagerly,
    a bare ``lax.scan`` re-lowers on every call — including its ``jvp`` and
    transpose rules — because the scan body jaxpr is rebuilt fresh each time and
    misses XLA's compilation cache.  Over a long optimisation loop that leaks
    compiled executables until the OS refuses an allocation mid-compile.
    Wrapping it here lowers the scan once per input shape.
    """
    return _rmf_twist_pure(gamma, gammadash)


# ============================================================================
# Base class
# ============================================================================

@jax.tree_util.register_pytree_node_class
class FramedCurveJAX:
    """Base class for pure-JAX framed curves.

    Attributes
    ----------
    curve : CurveJAX
        The underlying curve.
    alpha : jax.Array (nquad,)
        Rotation angles around the tangent at quadrature points
        (default: zeros).
    twist : jax.Array (nquad,) or None
        Frame-specific angle field built once at construction; ``None`` for
        frames with a closed-form pointwise evaluation (see
        :class:`FramedCurveRMFJAX`).  Carried as a pytree child so that JAX
        transformations never rebuild it.

    Subclasses implement :meth:`rotated_frame_eval`; everything else — the frame
    at quadpoints and its phi-derivative — is derived from it, so there is one
    frame definition per class.
    """

    def __init__(self, curve, alpha=None, *, twist=None):
        self.curve = curve
        n = curve.quadpoints.shape[0]
        if alpha is None:
            alpha = jnp.zeros(n)
        self.alpha = jnp.asarray(alpha, dtype=float)
        # None for frames with a closed-form pointwise evaluation; subclasses
        # that need stored state build it in their own __init__ when not
        # supplied.  It is a pytree child, so tree_unflatten hands the stored
        # value straight back and round-trips never rebuild.
        self.twist = twist

    def with_dofs(self, dofs):
        """Return a new framed curve of the same type built from new curve DOFs.

        Rebuilds the underlying :class:`~coil_fem.geo.CurveJAX` via
        :meth:`CurveJAX.with_dofs` and wraps it in the same frame subclass,
        preserving ``alpha``.  This is the differentiable entry point used to
        regenerate mesh geometry from traced DOFs: ``dofs`` is the only traced
        input; static curve metadata and ``alpha`` are treated as constants.
        The stored ``self.curve.dofs`` is intentionally ignored.

        ``twist`` is *not* carried over — new DOFs mean a new frame, so this is
        the single point at which frame construction happens per DOF update.

        Parameters
        ----------
        dofs : jax.Array
            Curve Fourier DOFs, same layout as ``self.curve.dofs``.

        Returns
        -------
        FramedCurveJAX
            A new instance of ``type(self)``.
        """
        return type(self)(self.curve.with_dofs(dofs), self.alpha)

    def alpha_eval(self, phi):
        r"""Analytic Fourier evaluation of ``alpha`` at arbitrary *phi*.

        Mirrors :meth:`CurveJAX.gamma_eval`: reconstructs the band-limited
        Fourier series of ``self.alpha`` and evaluates it at the supplied
        parameter values.

        This is *exact* for band-limited alpha
        (fewer than N/2 active modes) sampled at the uniform quadpoints
        ``phi_k = k/N``.

        Parameters
        ----------
        phi : array-like
            Target parameter values in [0, 1), arbitrary shape.

        Returns
        -------
        jnp.ndarray
            Shape ``phi.shape``. Reduces to :attr:`alpha` (to machine
            precision) when *phi* equals ``self.curve.quadpoints``.
        """
        phi_arr = jnp.asarray(phi, dtype=float)
        N = self.alpha.shape[0]
        alpha_hat = jnp.fft.rfft(self.alpha)              # (K,), K = N//2 + 1
        K = alpha_hat.shape[0]
        k = jnp.arange(K, dtype=float)                    # (K,)

        angle = 2.0 * jnp.pi * k * phi_arr[..., None]     # (..., K)
        real = (
            alpha_hat.real * jnp.cos(angle)
            - alpha_hat.imag * jnp.sin(angle)
        )                                                 # Re(α̂[k] e^{2πikφ})

        # Reconstruction weights: 1 for DC; 2 for general k; 1 for Nyquist
        # (only present when N is even).  Matches the band-limited periodic
        # interpolant that ``irfft`` produces at the sampling phases.
        if N % 2 == 0:
            weights = jnp.where((k == 0) | (k == K - 1), 1.0, 2.0)
        else:
            weights = jnp.where(k == 0, 1.0, 2.0)

        return jnp.sum(weights * real, axis=-1) / N

    def tree_flatten(self):
        return (self.curve, self.alpha, self.twist), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        curve, alpha, twist = children
        return cls(curve, alpha, twist=twist)

    # ============================================================================
    # Curve pass-throughs
    # ============================================================================

    def gamma(self):
        """Curve position at quadrature points, shape (nquad, 3)."""
        return self.curve.gamma()

    def gammadash(self):
        """First derivative at quadrature points, shape (nquad, 3)."""
        return self.curve.gammadash()

    def gammadashdash(self):
        """Second derivative at quadrature points, shape (nquad, 3)."""
        return self.curve.gammadashdash()

    def gamma_eval(self, phi):
        """Evaluate the curve at arbitrary parameter values *phi*.

        Equivalent to ``gamma()`` but evaluated at the supplied *phi*
        values instead of the stored quadrature points.

        Parameters
        ----------
        phi : array-like
            Curve parameter values in [0, 1), arbitrary shape.

        Returns
        -------
        jnp.ndarray
            Shape ``phi.shape + (3,)``.
        """
        return self.curve.gamma_eval(jnp.asarray(phi, dtype=float))

    # ============================================================================
    # Frame interface
    # ============================================================================

    def rotated_frame(self):
        """Return (t, p, q) at quadrature points, each shape (nquad, 3)."""
        return self.rotated_frame_eval(self.curve.quadpoints)

    def rotated_frame_and_dash_eval(self, phi):
        """Compute the frame and its phi-derivative at arbitrary *phi*.

        :meth:`rotated_frame_eval` is a smooth function of *phi*, so d/dphi is a
        plain JVP in *phi* — no chain rule through
        ``(gammadash, gammadashdash, d_alpha/d_phi)`` is needed, and the derivative
        is guaranteed to be that of the frame every other method returns.

        Parameters
        ----------
        phi : array-like
            Target parameter values, arbitrary shape.

        Returns
        -------
        (t, p, q) : tuple of jnp.ndarray, each shape ``phi.shape + (3,)``
            Frame vectors (primals).
        (tdash, pdash, qdash) : tuple of jnp.ndarray, each shape ``phi.shape + (3,)``
            Frame vector derivatives d/dphi (tangents).
        """
        phi = jnp.asarray(phi, dtype=float)
        return jax.jvp(self.rotated_frame_eval, (phi,), (jnp.ones_like(phi),))

    def _rotated_frame_and_dash(self):
        """Frame and d/dphi at the curve quadrature points."""
        return self.rotated_frame_and_dash_eval(self.curve.quadpoints)

    def rotated_frame_eval(self, phi):
        """Evaluate the rotated frame at arbitrary parameter values.

        Subclasses must override this with a pure-JAX implementation that
        is differentiable through ``self.curve.dofs``, ``self.alpha``, and
        ``phi``. See :class:`FramedCurveCentroidJAX` and
        :class:`FramedCurveRMFJAX`.

        Parameters
        ----------
        phi : array-like
            Target parameter values, arbitrary shape.

        Returns
        -------
        t, p, q : jnp.ndarray
            Each array has shape ``phi.shape + (3,)``.
        """
        raise NotImplementedError(
            "rotated_frame_eval is provided by subclasses "
            "(FramedCurveCentroidJAX, FramedCurveRMFJAX)."
        )

    # ============================================================================
    # Frame curvatures  κ₁, κ₂, κ₃
    # ============================================================================

    def frame_normal_curvature(self):
        r"""Frame normal curvature :math:`\kappa_1 = (d\mathbf{t}/dl)\cdot\mathbf{p}`.

        Returns the component of the curvature vector in the **p**
        direction.  Shape ``(nquad,)``.

        Here :math:`l` is arclength: :math:`d/dl = (1/|\gamma'|)\,d/d\phi`.
        """
        (_, p, _), (tdash, _, _) = self._rotated_frame_and_dash()
        gd_norm = jnp.linalg.norm(self.curve.gammadash(), axis=1)
        return jnp.sum(tdash * p, axis=1) / gd_norm

    def frame_binormal_curvature(self):
        r"""Frame binormal curvature :math:`\kappa_2 = (d\mathbf{t}/dl)\cdot\mathbf{q}`.

        Matches simsopt's ``frame_binormal_curvature()`` for the same
        frame type.  Shape ``(nquad,)``.
        """
        (_, _, q), (tdash, _, _) = self._rotated_frame_and_dash()
        gd_norm = jnp.linalg.norm(self.curve.gammadash(), axis=1)
        return jnp.sum(tdash * q, axis=1) / gd_norm

    def frame_torsion(self):
        r"""Frame torsion :math:`\kappa_3 = (d\mathbf{p}/dl)\cdot\mathbf{q}`.

        Matches simsopt's ``frame_torsion()`` for the same frame type.
        Shape ``(nquad,)``.
        """
        (_, p, q), (_, pdash, _) = self._rotated_frame_and_dash()
        gd_norm = jnp.linalg.norm(self.curve.gammadash(), axis=1)
        return jnp.sum(pdash * q, axis=1) / gd_norm

    def frame_curvatures_eval(self, phi):
        r"""Compute :math:`(\kappa_1, \kappa_2, \kappa_3)` at arbitrary *phi*.

        Parameters
        ----------
        phi : array-like
            Target parameter values, arbitrary shape.

        Returns
        -------
        kappa1, kappa2, kappa3 : jnp.ndarray, each shape ``phi.shape``
            Frame normal curvature, binormal curvature, and torsion.
        """
        (_, p, q), (tdash, pdash, _) = self.rotated_frame_and_dash_eval(phi)
        gd_norm = jnp.linalg.norm(self.curve.gamma_eval(phi, 1), axis=-1)
        kappa1 = jnp.sum(tdash * p, axis=-1) / gd_norm
        kappa2 = jnp.sum(tdash * q, axis=-1) / gd_norm
        kappa3 = jnp.sum(pdash * q, axis=-1) / gd_norm
        return kappa1, kappa2, kappa3

    def frame_curvatures(self):
        r"""Compute :math:`(\kappa_1, \kappa_2, \kappa_3)` at quadrature points.

        More efficient than calling :meth:`frame_normal_curvature`,
        :meth:`frame_binormal_curvature`, and :meth:`frame_torsion`
        separately when more than one is needed.

        Returns
        -------
        kappa1, kappa2, kappa3 : jnp.ndarray, each shape (nquad,)
            Frame normal curvature, binormal curvature, and torsion.
        """
        return self.frame_curvatures_eval(self.curve.quadpoints)

    def binorm(self):
        """Alias for :meth:`frame_binormal_curvature` (:math:`\\kappa_2`)."""
        return self.frame_binormal_curvature()

    def torsion(self):
        """Alias for :meth:`frame_torsion` (:math:`\\kappa_3`)."""
        return self.frame_torsion()


# ============================================================================
# Centroid frame
# ============================================================================

@jax.tree_util.register_pytree_node_class
class FramedCurveCentroidJAX(FramedCurveJAX):
    """Centroid frame in pure JAX.

    The frame introduced in Singh et al., "Optimization of finite-build
    stellarator coils", J. Plasma Phys. 86 (2020).

    Frame vectors: :math:`\\mathbf{t}` is normalised ``gammadash``;
    :math:`\\mathbf{p}` is ``(gamma - centroid)`` projected perpendicular to
    :math:`\\mathbf{t}`, normalised; :math:`\\mathbf{q} = \\mathbf{t} \\times \\mathbf{p}`.
    Then the frame is rotated by *alpha* around :math:`\\mathbf{t}`.
    """

    def rotated_frame_eval(self, phi):
        r"""Closed-form centroid frame at arbitrary parameter values.

        Pure JAX, fully differentiable through ``self.curve.dofs``,
        ``self.alpha``, and ``phi``.  Uses :meth:`CurveJAX.gamma_eval`
        and :meth:`alpha_eval`, with the centroid taken as the discrete
        mean over the curve's quadpoints, ``jnp.mean(self.curve.gamma(), axis=0)``.

        Parameters
        ----------
        phi : array-like
            Target parameter values, arbitrary shape.

        Returns
        -------
        t, p, q : jnp.ndarray
            Each array has shape ``phi.shape + (3,)``.
        """
        phi_arr = jnp.asarray(phi, dtype=float)
        t, p0, q0 = _centroid_reference(
            self.curve.gamma_eval(phi_arr, 0),
            self.curve.gamma_eval(phi_arr, 1),
            jnp.mean(self.curve.gamma(), axis=0),
        )
        alpha = self.alpha_eval(phi_arr)
        ca = jnp.cos(alpha)[..., None]
        sa = jnp.sin(alpha)[..., None]
        return t, ca * p0 - sa * q0, sa * p0 + ca * q0


# ============================================================================
# Rotation-minimizing frame
# ============================================================================

@jax.tree_util.register_pytree_node_class
class FramedCurveRMFJAX(FramedCurveJAX):
    """Rotation-minimizing frame (RMF / Bishop frame) in pure JAX.

    The frame is constructed with the double-reflection algorithm of
    Wang et al. (2008) "Computation of rotation minimizing frames",
    ACM Transactions on Graphics 27(1).

    The RMF minimises the twist of the normal along the curve, making
    it ideal for circular or near-circular cross-sections.  A uniform
    angular correction is applied to enforce periodicity for closed
    curves; the residual gap scales as O(1/N).

    Frame vectors: :math:`\\mathbf{t}` is normalised ``gammadash``;
    :math:`\\mathbf{p}` is the RMF first transverse vector (twist-minimising);
    :math:`\\mathbf{q} = \\mathbf{t} \\times \\mathbf{p}`.
    Then the frame is rotated by *alpha* around :math:`\\mathbf{t}`.
    """

    def __init__(self, curve, alpha=None, *, twist=None):
        # The RMF has no closed-form pointwise expression, so the propagation is
        # run here — once — and stored as :attr:`twist`.  Skipped when supplied,
        # which is how tree_unflatten avoids re-running the scan.
        if twist is None:
            twist = _twist_jitted(curve.gamma(), curve.gammadash())
        super().__init__(curve, alpha, twist=twist)

    def rotated_frame_eval(self, phi):
        r"""RMF at arbitrary parameter values, from the twist built at construction.

        The RMF is defined by discrete propagation along the curve, so it has
        no closed-form pointwise expression.  Instead the propagation is run
        **once** (in ``__init__``, giving :attr:`twist`) and evaluation
        interpolates that angle field against the closed-form centroid
        reference frame.  Consequences:

        * ``t`` is exact — it comes from :meth:`CurveJAX.gamma_eval`,
          never from interpolation.
        * ``p`` and ``q`` are exactly orthonormal by construction, because only
          a scalar angle is interpolated.
        * The result is smooth in *phi* and independent of the ordering or
          density of the query points, so scattered *phi* is fully supported.

        Parameters
        ----------
        phi : array-like
            Target parameter values, arbitrary shape.

        Returns
        -------
        t, p, q : jnp.ndarray
            Each array has shape ``phi.shape + (3,)``.
        """
        phi_arr = jnp.asarray(phi, dtype=float)
        gamma = self.curve.gamma_eval(phi_arr, 0)
        gammadash = self.curve.gamma_eval(phi_arr, 1)
        t, p_c, q_c = _centroid_reference(
            gamma, gammadash, jnp.mean(self.curve.gamma(), axis=0)
        )

        periodic, total = _unwrap_periodic(self.twist)
        theta = _trig_interp(periodic, phi_arr) + total * phi_arr
        angle = theta + self.alpha_eval(phi_arr)

        ca = jnp.cos(angle)[..., None]
        sa = jnp.sin(angle)[..., None]
        return t, ca * p_c - sa * q_c, sa * p_c + ca * q_c


# ============================================================================
# Convenience constructors
# ============================================================================

def _to_jax_curve(curve):
    """Accept a :class:`CurveJAX` or a simsopt ``CurveXYZFourier`` / ``CurveRZFourier``."""
    from .curve_jax import curve_jax_from_simsopt
    return curve_jax_from_simsopt(curve)


def make_centroid_frame(curve, alpha=None):
    """Create a centroid-framed curve.

    Parameters
    ----------
    curve : CurveJAX or simsopt CurveXYZFourier / CurveRZFourier
    alpha : array-like, optional
        Rotation angles at quadrature points (default: zeros).
        ``d(alpha)/d(phi)`` is derived automatically via FFT differentiation.

    Returns
    -------
    FramedCurveCentroidJAX
    """
    return FramedCurveCentroidJAX(_to_jax_curve(curve), alpha)


def make_rmf_frame(curve, alpha=None):
    """Create an RMF-framed curve.

    Parameters
    ----------
    curve : CurveJAX or simsopt CurveXYZFourier / CurveRZFourier
    alpha : array-like, optional
        Rotation angles at quadrature points (default: zeros).
        ``d(alpha)/d(phi)`` is derived automatically via FFT differentiation.

    Returns
    -------
    FramedCurveRMFJAX
    """
    return FramedCurveRMFJAX(_to_jax_curve(curve), alpha)


def make_framed_curve(curve, frame_type, alpha=None):
    """Dispatch a frame-type string to the matching framed-curve class.

    Parameters
    ----------
    curve : CurveJAX or simsopt CurveXYZFourier / CurveRZFourier
    frame_type : {'rmf', 'centroid'}
        ``'rmf'`` for a rotation-minimising frame, ``'centroid'`` for the
        centroid frame.
    alpha : array-like, optional
        Rotation angles at quadrature points (default: zeros).

    Returns
    -------
    FramedCurveRMFJAX or FramedCurveCentroidJAX
    """
    if frame_type == 'rmf':
        return make_rmf_frame(curve, alpha)
    elif frame_type == 'centroid':
        return make_centroid_frame(curve, alpha)
    else:
        raise ValueError(
            f"frame_type must be 'rmf' or 'centroid', got {frame_type!r}."
        )
