"""Magnetic field computation for stellarator coils.

Provides :func:`biot_savart` (filament Biot-Savart kernel),
:func:`B_self_quadrature` (self-field at FEM quadrature points via the
Landreman-Hurwitz-Antonsen 2025 formula for rectangular cross-sections), and
:func:`lorentz_body_force` (``J × B`` body-force density).

Cross-section coordinate convention (LHA 2025)
-----------------------------------------------
Points inside the cross-section are parametrised as
``r(φ, u, v) = r_c(φ) + (u·a/2)·p(φ) + (v·b/2)·q(φ)``,
where ``a = w1``, ``b = w2``, and ``u, v ∈ [-1, 1]``.

References
----------
Landreman, Hurwitz & Antonsen, Nucl. Fusion 65, 036008 (2025)
"""

from __future__ import annotations
# NOTE: simsopt's process-wide jax_platform_name="cpu" pin is cleared in
# coil_fem.gpu_env, invoked from coil_fem/__init__.py after this import.
from simsopt.field.selffield import B_regularized_pure
import os
import jax
import jax.numpy as jnp
from jax import vmap


# mu_0 / (4 pi)
_BIOT_SAVART_PREFACTOR = 1e-7


# ============================================================================
# LHA rectangular cross-section helpers
# ============================================================================

def _rect_k(a, b):
    r"""Auxiliary function k for a rectangular cross-section (Eq. 13).

    .. math::

        k = \frac{4b}{3a}\arctan\frac{a}{b} + \frac{4a}{3b}\arctan\frac{b}{a}
          + \frac{b^2}{6a^2}\ln\frac{b}{a} + \frac{a^2}{6b^2}\ln\frac{a}{b}
          - \frac{a^4 - 6a^2b^2 + b^4}{6a^2b^2}
            \ln\!\left(\frac{a}{b} + \frac{b}{a}\right)

    Parameters
    ----------
    a, b : scalar JAX arrays — cross-section full-widths (a = w1, b = w2)
    """
    return (
        (4.0 * b / (3.0 * a)) * jnp.arctan(a / b)
        + (4.0 * a / (3.0 * b)) * jnp.arctan(b / a)
        + (b**2 / (6.0 * a**2)) * jnp.log(b / a)
        + (a**2 / (6.0 * b**2)) * jnp.log(a / b)
        - (a**4 - 6.0 * a**2 * b**2 + b**4) / (6.0 * a**2 * b**2)
        * jnp.log(a / b + b / a)
    )



def _G_rect(x, y):
    r"""Auxiliary function G from Eq. (18).

    .. math::

        G(x, y) = y\,\arctan\frac{x}{y}
                + \frac{x}{2}\ln\!\left(1 + \frac{y^2}{x^2}\right)

    Works element-wise for scalar or array inputs.  Both terms vanish in the
    limit as their prefactor does, but ``y / x`` overflows for ``x == 0``, so
    the naive form evaluates ``0 * inf`` and returns NaN.  ``x == 0`` occurs
    whenever a query point lies exactly on a conductor face (``|u| == 1`` or
    ``|v| == 1``), which happens when the field is sampled at mesh nodes rather
    than interior quadrature points.  The double-``where`` guard returns the
    correct limits and keeps gradients finite.
    """
    x_zero = x == 0
    y_zero = y == 0
    x_safe = jnp.where(x_zero, 1.0, x)
    y_safe = jnp.where(y_zero, 1.0, y)
    return (
        jnp.where(y_zero, 0.0, y * jnp.arctan(x / y_safe))
        + jnp.where(x_zero, 0.0, 0.5 * x * jnp.log1p((y / x_safe) ** 2))
    )


def _K_rect_flat(U, V, a, b, kappa1, kappa2, p, q):
    r"""K function from Eq. (20), vectorised over a flat set of points.

    All per-point arrays share a single leading axis ``n_pts``.

    Parameters
    ----------
    U, V : (n_pts,) arrays — corner offsets (u - s_u, v - s_v) at each point
    a, b : scalar — full cross-section widths
    kappa1, kappa2 : (n_pts,)
    p, q : (n_pts, 3)

    Returns
    -------
    (n_pts, 3)
    """
    # Promote to (n_pts, 1) so they broadcast with (n_pts, 3) frame vectors.
    U1 = U[:, None]
    V1 = V[:, None]

    S  = a * U1**2 / b + b * V1**2 / a      # (n_pts, 1)

    # S vanishes only at a cross-section corner (U == V == 0), where every term
    # below tends to zero but evaluates as 0 * inf.  Guard the log and the two
    # arctan denominators, then return the limit.  Away from the corner a zero
    # denominator is harmless: arctan saturates at +/- pi/2 as intended.
    at_corner = S == 0.0
    S_safe = jnp.where(at_corner, 1.0, S)
    U_safe = jnp.where(at_corner, 1.0, U1)
    V_safe = jnp.where(at_corner, 1.0, V1)
    L  = jnp.log(S_safe)                     # (n_pts, 1)

    k1q_m_k2p = kappa1[:, None] * q - kappa2[:, None] * p   # (n_pts, 3)
    k2q_m_k1p = kappa2[:, None] * q - kappa1[:, None] * p

    # Eq. (20) line 1: -2UV (kappa1 q - kappa2 p) ln S
    term1 = -2.0 * U1 * V1 * L * k1q_m_k2p
    # Eq. (20) line 2: (kappa2 q - kappa1 p) S ln S
    term2 = S * L * k2q_m_k1p
    # Eq. (20) line 3: (4 a U^2 / b) kappa2 p arctan(bV / (aU))
    atan_bV_aU = jnp.arctan(b * V1 / (a * U_safe))
    term3 = (4.0 * a * U1**2 / b) * atan_bV_aU * kappa2[:, None] * p
    # Eq. (20) line 4: -(4 b V^2 / a) kappa1 q arctan(aU / (bV))
    atan_aU_bV = jnp.arctan(a * U1 / (b * V_safe))
    term4 = -(4.0 * b * V1**2 / a) * atan_aU_bV * kappa1[:, None] * q

    return jnp.where(at_corner, 0.0, term1 + term2 + term3 + term4)


# ============================================================================
# Biot-Savart kernel
# ============================================================================

def biot_savart(
    target_points: jax.Array,
    source_gammas: jax.Array,
    source_gammadashs: jax.Array,
    source_currents: jax.Array,
) -> jax.Array:
    """Biot-Savart field at arbitrary target points from filament sources.

    All operations are pure JAX and fully differentiable through both
    ``target_points`` (mesh node/quadrature positions) and
    ``source_gammas`` / ``source_gammadashs`` (coil DOFs).

    To exclude the self-contribution, zero the relevant current before calling:
    ``source_currents.at[coil_idx].set(0.0)``.

    Parameters
    ----------
    target_points : (n_targets, 3)
    source_gammas : (n_src, n_quad, 3)
    source_gammadashs : (n_src, n_quad, 3)
        Unnormalised tangent vectors d(gamma)/d(phi).
    source_currents : (n_src,) [A]

    Returns
    -------
    (n_targets, 3) [T]

    Notes
    -----
    Discrete Biot-Savart::

        B = (mu_0 / 4pi) * (1 / n_quad)
            * sum_j I_j * sum_q (dl_q x r_pq) / |r_pq|^3

    where ``r_pq = target_p - source_q`` and ``dl_q = gammadash_q * dphi``.
    """
    n_quad = source_gammas.shape[1]

    def _one_source(B_acc, args):
        gamma_j, gammadash_j, I_j = args

        def at_one_target(pt):
            r = pt[None, :] - gamma_j                       # (n_quad, 3)
            r_norm = jnp.sqrt(
                jnp.sum(r * r, axis=-1, keepdims=True) + 1e-30
            )
            r_norm_safe = jnp.where(r_norm < 1e-14, 1e-14, r_norm)
            cross = jnp.cross(gammadash_j, r)               # (n_quad, 3)
            dB = cross / r_norm_safe ** 3
            return I_j * jnp.sum(dB, axis=0)               # (3,)

        return B_acc + vmap(at_one_target)(target_points), None  # (n_targets, 3)

    B, _ = jax.lax.scan(
        _one_source,
        jnp.zeros_like(target_points),
        (source_gammas, source_gammadashs, source_currents),
    )
    return _BIOT_SAVART_PREFACTOR / n_quad * B


# ============================================================================
# Self-field at FEM quadrature points
# ============================================================================

def B_self_quadrature(
    framed_curve,
    current: jax.Array | float,
    cross_section: dict,
    phi_quad: jax.Array,
    uv_quad: jax.Array | None,
) -> jax.Array:
    r"""Self-field at FEM quadrature points.

    Implements the full Landreman-Hurwitz-Antonsen (2025) formula
    **B = B_reg + B_0 + B_kappa + B_b** for rectangular cross-sections
    evaluated at arbitrary (phi, u, v) positions inside the conductor.

    Parameters
    ----------
    framed_curve : FramedCurveJAX
        Framed coil curve (centroid or RMF frame).
    current : scalar [A]
    cross_section : dict
        ``{'shape': 'rect', 'w1': a, 'w2': b}`` or
        ``{'shape': 'disk', 'radius': r}``.
        *w1* and *w2* are the **full** conductor widths (a = w1, b = w2
        in the LHA 2025 notation).
    phi_quad : (n_cells, n_quads) array
        Curve parameter phi at each FEM quadrature point.  Values outside
        ``[0, 1)`` at the periodic seam are handled via ``period=1.0``.
    uv_quad : (n_cells, n_quads, 2) array or None
        Cross-section coordinates (u, v) in [-1, 1] at each FEM quadrature
        point.  Required for ``shape='rect'``; ignored for ``'disk'`` (which
        raises ``NotImplementedError`` regardless).

    Returns
    -------
    (n_cells, n_quads, 3) [T]

    Raises
    ------
    NotImplementedError
        When ``cross_section['shape'] == 'disk'``.  A closed-form self-field
        analogous to the LHA (2025) formula for circular cross-sections has
        not yet been implemented.  See the coil-fem issue tracker.
    """
    import interpax

    shape = cross_section['shape']
    if shape == 'disk':
        raise NotImplementedError(
            "B_self_quadrature: self-field for disk (circular) cross-sections "
            "is not yet implemented.  The formula analogous to Landreman-"
            "Hurwitz-Antonsen (2025) for circular cross-sections is known "
            "but has not yet been coded here."
        )
    if shape != 'rect':
        raise ValueError(
            f"B_self_quadrature: unknown cross_section shape {shape!r}. "
            "Expected 'rect' or 'disk'."
        )

    a = jnp.asarray(cross_section['w1'], dtype=float)   # full width in p dir
    b = jnp.asarray(cross_section['w2'], dtype=float)   # full width in q dir
    I = jnp.asarray(current, dtype=float).reshape(())

    # ---------- Curve quantities at centerline quadrature points ----------
    curve         = framed_curve.curve
    gamma         = curve.gamma()           # (n_phi, 3)
    gammadash     = curve.gammadash()
    gammadashdash = curve.gammadashdash()
    quadpoints    = curve.quadpoints        # (n_phi,) uniform in [0, 1)

    _, p_cl, q_cl = framed_curve.rotated_frame()               # each (n_phi, 3)
    kappa1_cl, kappa2_cl, _ = framed_curve.frame_curvatures()  # each (n_phi,)

    # ------------------------------------------------------------------
    # delta and regularization  (Eqs. 12-13)
    # \delta = \exp\!\left(-\tfrac{25}{6} + k(a, b)\right)
    # The product ``delta * a * b`` equals ``regularization_rect(a, b)`` from
    # simsopt.
    # ------------------------------------------------------------------
    delta = jnp.exp(-25.0 / 6.0 + _rect_k(a, b))  
    reg   = delta * a * b

    # ------------------------------------------------------------------
    # B_reg at centerline quadpoints  (Eq. 15)
    # ------------------------------------------------------------------
    B_reg_cl = B_regularized_pure(
        gamma, gammadash, gammadashdash, quadpoints, I, reg
    )  # (n_phi, 3)

    # ------------------------------------------------------------------
    # B_b at centerline quadpoints  (Eq. 21)
    # B_b = (mu_0 I / 8pi) (4 + 2 ln2 + ln delta) (kappa1 q - kappa2 p)
    # mu_0 I / (8 pi) = _BIOT_SAVART_PREFACTOR * I / 2
    # ------------------------------------------------------------------
    log_delta  = jnp.log(delta)
    b_coeff    = 4.0 + 2.0 * jnp.log(2.0) + log_delta
    kappa_b_cl = kappa1_cl[:, None] * q_cl - kappa2_cl[:, None] * p_cl  # (n_phi,3)
    B_b_cl     = 0.5 * _BIOT_SAVART_PREFACTOR * I * b_coeff * kappa_b_cl

    # ------------------------------------------------------------------
    # Interpolate phi-dependent quantities to FEM quad points via
    # periodic C2 cubic splines.  phi_quad may contain values > 1 at
    # the periodic seam; interpax wraps them via period=1.0.
    # ------------------------------------------------------------------
    n_cells, n_quads_per_cell = phi_quad.shape
    phi_flat = phi_quad.ravel()   # (n_pts,)

    def _interp(y):
        """Lift (n_phi, ...) -> (n_pts, ...) via C2 cubic periodic spline."""
        return interpax.interp1d(
            phi_flat, quadpoints, y, method='cubic2', period=1.0
        )

    B_reg_q  = _interp(B_reg_cl)   # (n_pts, 3)
    B_b_q    = _interp(B_b_cl)     # (n_pts, 3)
    p_q      = _interp(p_cl)       # (n_pts, 3)
    q_q      = _interp(q_cl)       # (n_pts, 3)
    kappa1_q = _interp(kappa1_cl)  # (n_pts,)
    kappa2_q = _interp(kappa2_cl)  # (n_pts,)

    n_pts = n_cells * n_quads_per_cell
    u_flat = uv_quad[..., 0].ravel()  # (n_pts,)
    v_flat = uv_quad[..., 1].ravel()  # (n_pts,)

    # ------------------------------------------------------------------
    # B_0  (Eq. 17)
    # B_0 = (mu_0 I / (4pi * a * b)) *
    #   sum_{su,sv=+/-1} su*sv [G(b(v-sv), a(u-su)) q - G(a(u-su), b(v-sv)) p]
    # Prefactor: _BIOT_SAVART_PREFACTOR * I / (a * b)
    # ------------------------------------------------------------------
    B_0_q = jnp.zeros((n_pts, 3))
    for su in (1.0, -1.0):
        for sv in (1.0, -1.0):
            U = u_flat - su    # (n_pts,)
            V = v_flat - sv
            G_bV_aU = _G_rect(b * V, a * U)   # (n_pts,)
            G_aU_bV = _G_rect(a * U, b * V)   # (n_pts,)
            B_0_q = B_0_q + su * sv * (
                G_bV_aU[:, None] * q_q - G_aU_bV[:, None] * p_q
            )
    B_0_q = (_BIOT_SAVART_PREFACTOR * I / (a * b)) * B_0_q

    # ------------------------------------------------------------------
    # B_kappa  (Eqs. 19-20)
    # Prefactor: mu_0 I / (64 pi) = _BIOT_SAVART_PREFACTOR * I / 16
    # ------------------------------------------------------------------
    B_kappa_q = jnp.zeros((n_pts, 3))
    for su in (1.0, -1.0):
        for sv in (1.0, -1.0):
            U = u_flat - su   # (n_pts,)
            V = v_flat - sv
            K_val = _K_rect_flat(U, V, a, b, kappa1_q, kappa2_q, p_q, q_q)
            B_kappa_q = B_kappa_q + su * sv * K_val
    B_kappa_q = (_BIOT_SAVART_PREFACTOR / 16.0) * I * B_kappa_q

    # ------------------------------------------------------------------
    # Sum and reshape to (n_cells, n_quads_per_cell, 3)
    # ------------------------------------------------------------------
    B_total = B_reg_q + B_0_q + B_kappa_q + B_b_q
    return B_total.reshape(n_cells, n_quads_per_cell, 3)


# ============================================================================
# Lorentz body force
# ============================================================================

def lorentz_body_force(J: jax.Array, B: jax.Array) -> jax.Array:
    """Lorentz body force density **f = J × B**.

    The exact law is ``f = J × B``.  For a coil with uniform current ``I``
    distributed over cross-section area ``A`` with tangent ``t_hat``,
    ``J = (I / A) t_hat`` — but the caller is responsible for building ``J``,
    so this stays agnostic to the current model (uniform, cable, skin-effect,
    ...).

    Parameters
    ----------
    J : jax.Array, shape ``(..., 3)``
        Current density [A/m²].
    B : jax.Array, shape ``(..., 3)``
        Magnetic field [T].

    Returns
    -------
    jax.Array, shape ``(..., 3)`` [N/m³]
    """
    return jnp.cross(J, B)
