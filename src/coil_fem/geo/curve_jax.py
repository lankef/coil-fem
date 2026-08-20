"""Pure-JAX curve pytrees with a shared differential-geometry API.

:class:`CurveJAX` provides ``gamma``, derivatives, ``kappa``, ``torsion``,
``incremental_arclength``, and ``frenet_frame``.  Concrete subclasses
(:class:`CurveXYZFourierJAX`, :class:`CurveRZFourierJAX`) implement
representation-specific ``gamma_eval`` and simsopt interop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ============================================================================
# CurveJAX base
# ============================================================================

class CurveJAX:
    """Shared differential-geometry API for JAX curve pytrees.

    Subclasses must implement ``gamma_eval``, ``get_dofs``, ``from_simsopt``,
    ``to_simsopt``, ``tree_flatten``, ``tree_unflatten``, and ``curve_center``.
    Pytree children are ``(quadpoints, dofs, *rest)`` so :meth:`with_dofs`
    works without per-subclass overrides.
    """

    def with_dofs(self, dofs):
        """Return a new curve of the same type with replaced DOFs.

        Parameters
        ----------
        dofs : jax.Array
            New coefficient vector, same layout as ``self.dofs``.

        Returns
        -------
        CurveJAX
            Fresh instance of ``type(self)`` with the same static metadata.
        """
        children, aux = self.tree_flatten()
        return type(self).tree_unflatten(
            aux, (children[0], dofs) + tuple(children[2:])
        )

    def gamma_eval(self, phi, diff_order: int = 0):
        """Evaluate the curve or its derivative at arbitrary parameter values.

        Parameters
        ----------
        phi : array-like
            Curve parameter values in ``[0, 1)``, arbitrary shape.
        diff_order : int
            Derivative order (0 = positions, 1 = first derivative, …).

        Returns
        -------
        jax.Array, shape ``phi.shape + (3,)``
        """
        raise NotImplementedError

    def get_dofs(self):
        """Return the flat DOF vector (simsopt-compatible)."""
        raise NotImplementedError

    def curve_center(self):
        """Return a representative center point, shape ``(3,)``."""
        raise NotImplementedError

    def gamma(self):
        """Curve position, shape (nquad, 3)."""
        return self.gamma_eval(self.quadpoints, 0)

    def gammadash(self):
        """First derivative d(gamma)/d(phi), shape (nquad, 3)."""
        return self.gamma_eval(self.quadpoints, 1)

    def gammadashdash(self):
        """Second derivative d2(gamma)/d(phi)2, shape (nquad, 3)."""
        return self.gamma_eval(self.quadpoints, 2)

    def gammadashdashdash(self):
        """Third derivative d3(gamma)/d(phi)3, shape (nquad, 3)."""
        return self.gamma_eval(self.quadpoints, 3)

    def kappa(self):
        r"""Curvature :math:`\kappa = \|\gamma' \times \gamma''\| / \|\gamma'\|^3`.

        Returns
        -------
        jax.Array, shape (nquad,)
        """
        d1 = self.gammadash()
        d2 = self.gammadashdash()
        return (
            jnp.linalg.norm(jnp.cross(d1, d2), axis=1)
            / jnp.linalg.norm(d1, axis=1) ** 3
        )

    def torsion(self):
        r"""Torsion :math:`\tau = (\gamma' \times \gamma'') \cdot \gamma''' / \|\gamma' \times \gamma''\|^2`.

        Returns
        -------
        jax.Array, shape (nquad,)
        """
        d1 = self.gammadash()
        d2 = self.gammadashdash()
        d3 = self.gammadashdashdash()
        cross = jnp.cross(d1, d2)                        # (nquad, 3)
        return jnp.sum(cross * d3, axis=1) / jnp.sum(cross ** 2, axis=1)

    def incremental_arclength(self):
        r"""Incremental arclength :math:`ds = \|\gamma'(\phi)\| d\phi` at each quadrature point.

        Returns
        -------
        jax.Array, shape (nquad,)
            :math:`\|\gamma'(\phi)\|` at each quadrature point.
        """
        d1 = self.gammadash()
        return jnp.linalg.norm(d1, axis=1)

    def frenet_frame(self):
        r"""
        Frenet-Serret frame :math:`(\mathbf{t}, \mathbf{n}, \mathbf{b})` at each quadrature point.

        .. math::

            \mathbf{t} = \frac{d\boldsymbol{\gamma}/d\phi}{|d\boldsymbol{\gamma}/d\phi|},\quad
            \mathbf{n} = \frac{d\mathbf{t}/ds}{|d\mathbf{t}/ds|},\quad
            \mathbf{b} = \mathbf{t} \times \mathbf{n}

        where :math:`s` is arclength.

        Returns
        -------
        tuple of jax.Array
            ``(t, n, b)``, each with shape ``(nquad, 3)``.
        """
        d1 = self.gammadash()
        d2 = self.gammadashdash()

        arclength = jnp.linalg.norm(d1, axis=1)          # (nquad,)

        t = d1 / arclength[:, None]

        # Derivative of t w.r.t. phi (chain rule on gamma'/|gamma'|):
        #   tdash = gamma''/|gamma'| - (gamma'·gamma'' / |gamma'|^3) * gamma'
        inner_d1_d2 = jnp.sum(d1 * d2, axis=1)           # (nquad,)
        tdash = (
            d2 / arclength[:, None]
            - (inner_d1_d2 / arclength ** 3)[:, None] * d1
        )

        n = tdash / jnp.linalg.norm(tdash, axis=1)[:, None]
        b = jnp.cross(t, n)
        return t, n, b


# ============================================================================
# CurveXYZFourierJAX
# ============================================================================

@jax.tree_util.register_pytree_node_class
class CurveXYZFourierJAX(CurveJAX):
    r"""JAX pytree implementation of ``CurveXYZFourier``.

    The curve is parameterised by :math:`\phi \in [0, 1)` and represented as

    .. math::

        x(\phi) = x_{c,0} + \sum_{j=1}^{\mathrm{order}}
            \bigl( x_{s,j} \sin(2\pi j\phi) + x_{c,j} \cos(2\pi j\phi) \bigr)

    and analogously for :math:`y(\phi)` and :math:`z(\phi)`.

    Parameters
    ----------
    quadpoints :
        1-D array of sample points in ``[0, 1)``, shape ``(nquad,)``.
    dofs :
        Fourier coefficients, shape ``(3*(2*order+1),)``, stored as
        ``[xc0, xs1, xc1, ..., xs_order, xc_order, yc0, ..., zc0, ...]``.
    order :
        Maximum Fourier mode number (static).
    """

    def __init__(self, quadpoints, dofs, order: int):
        self.quadpoints = jnp.asarray(quadpoints, dtype=float)
        self.dofs = jnp.asarray(dofs, dtype=float)
        self.order = order  # static -- lives in aux_data, not in leaves

    def tree_flatten(self):
        children = (self.quadpoints, self.dofs)
        aux_data = self.order
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        quadpoints, dofs = children
        return cls(quadpoints, dofs, aux_data)

    @classmethod
    def from_simsopt(cls, curve):
        """Construct a ``CurveXYZFourierJAX`` from a simsopt ``CurveXYZFourier``.

        The DOF layout is identical between the two representations
        (``[c0, s1, c1, …]`` per coordinate), so no reordering is needed.
        """
        return cls(
            quadpoints=curve.quadpoints,
            dofs=curve.get_dofs(),
            order=curve.order,
        )

    def to_simsopt(self):
        """Convert to a simsopt ``CurveXYZFourier`` with the same quadpoints, order, and DOFs."""
        import numpy as np
        from simsopt.geo import CurveXYZFourier

        curve = CurveXYZFourier(np.asarray(self.quadpoints), self.order)
        curve.set_dofs(np.asarray(self.dofs))
        return curve

    def get_dofs(self):
        """For compatibility with functions that consume simsopt Curves."""
        return self.dofs

    def curve_center(self):
        """Return ``[xc(0), yc(0), zc(0)]`` from the packed XYZ DOFs."""
        k = 2 * self.order + 1
        dofs = self.get_dofs()
        return dofs[jnp.array([0, k, 2 * k])]

    def gamma_eval(self, phi, diff_order: int = 0):
        """Evaluate the curve or its derivative at arbitrary parameter values.

        Parameters
        ----------
        phi : array-like
            Curve parameter values in ``[0, 1)``, arbitrary shape.
        diff_order : int
            Derivative order (0 = positions, 1 = first derivative, …).

        Returns
        -------
        jax.Array, shape ``phi.shape + (3,)``
        """
        phi = jnp.asarray(phi, dtype=float)
        phi_shape = phi.shape
        phi_flat = phi.ravel()                                        # (N,)

        k = 2 * self.order + 1
        jrange = jnp.arange(1, self.order + 1)                       # (order,)
        angle = 2.0 * jnp.pi * jrange[:, None] * phi_flat[None, :]  # (order, N)

        phase = diff_order * jnp.pi / 2.0
        twopij_n = (2.0 * jnp.pi * jrange) ** diff_order             # (order,)
        sin_basis = twopij_n[:, None] * jnp.sin(angle + phase)       # (order, N)
        cos_basis = twopij_n[:, None] * jnp.cos(angle + phase)       # (order, N)

        coords = []
        for i in range(3):
            # Constant term contributes only at diff_order == 0
            c = self.dofs[i * k] if diff_order == 0 else 0.0
            c = c + jnp.sum(
                self.dofs[i * k + 2 * jrange - 1, None] * sin_basis
                + self.dofs[i * k + 2 * jrange,     None] * cos_basis,
                axis=0,
            )
            coords.append(c)

        result = jnp.stack(coords, axis=-1)                           # (N, 3)
        return result.reshape(phi_shape + (3,))


# ============================================================================
# CurveRZFourierJAX
# ============================================================================

@jax.tree_util.register_pytree_node_class
class CurveRZFourierJAX(CurveJAX):
    r"""JAX pytree implementation of simsopt ``CurveRZFourier``.

    Cylindrical Fourier series with field-period harmonics, converted to
    Cartesian coordinates:

    .. math::

        r(\phi) &= \sum_{m=0}^{\mathrm{order}} r_{c,m}\cos(n_{\mathrm{fp}} m\,\theta)
                 + \sum_{m=1}^{\mathrm{order}} r_{s,m}\sin(n_{\mathrm{fp}} m\,\theta) \\
        z(\phi) &= \sum_{m=0}^{\mathrm{order}} z_{c,m}\cos(n_{\mathrm{fp}} m\,\theta)
                 + \sum_{m=1}^{\mathrm{order}} z_{s,m}\sin(n_{\mathrm{fp}} m\,\theta)

    with :math:`\theta = 2\pi\phi` and :math:`(x, y) = (r\cos\theta, r\sin\theta)`.
    When ``stellsym=True``, the ``rs`` and ``zc`` coefficients are zero and omitted
    from the DOF vector.

    Parameters
    ----------
    quadpoints :
        1-D array of sample points in ``[0, 1)``, shape ``(nquad,)``.
    dofs :
        Flat coefficient vector.  With ``stellsym=True``:
        ``[rc_0..rc_order, zs_1..zs_order]`` (length ``2*order+1``).
        With ``stellsym=False``:
        ``[rc_0..rc_order, rs_1..rs_order, zc_0..zc_order, zs_1..zs_order]``
        (length ``4*order+2``).
    order :
        Maximum Fourier mode number (static).
    nfp :
        Number of field periods (static).
    stellsym :
        If ``True``, stellarator symmetry (static).
    """

    def __init__(self, quadpoints, dofs, order: int, nfp: int, stellsym: bool):
        self.quadpoints = jnp.asarray(quadpoints, dtype=float)
        self.dofs = jnp.asarray(dofs, dtype=float)
        self.order = order
        self.nfp = nfp
        self.stellsym = stellsym

    def tree_flatten(self):
        children = (self.quadpoints, self.dofs)
        aux_data = (self.order, self.nfp, self.stellsym)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        quadpoints, dofs = children
        order, nfp, stellsym = aux_data
        return cls(quadpoints, dofs, order, nfp, stellsym)

    @classmethod
    def from_simsopt(cls, curve):
        """Construct from a simsopt ``CurveRZFourier``."""
        return cls(
            quadpoints=curve.quadpoints,
            dofs=curve.get_dofs(),
            order=curve.order,
            nfp=curve.nfp,
            stellsym=curve.stellsym,
        )

    def to_simsopt(self):
        """Convert to a simsopt ``CurveRZFourier`` with matching DOFs."""
        import numpy as np
        from simsopt.geo import CurveRZFourier

        curve = CurveRZFourier(
            np.asarray(self.quadpoints),
            self.order,
            self.nfp,
            self.stellsym,
        )
        curve.set_dofs(np.asarray(self.dofs))
        return curve

    def get_dofs(self):
        """For compatibility with functions that consume simsopt Curves."""
        return self.dofs

    def curve_center(self):
        """Return the origin ``(0, 0, 0)`` (RZ curves are centred on the z-axis)."""
        return jnp.zeros(3)

    def _gamma_pos(self, phi):
        """Cartesian positions for arbitrary ``phi`` (no derivatives)."""
        phi = jnp.asarray(phi, dtype=float)
        phi_shape = phi.shape
        phi_flat = phi.ravel()
        angle = 2.0 * jnp.pi * phi_flat                      # (N,)
        order = self.order
        nfp = self.nfp
        dofs = self.dofs

        m = jnp.arange(order + 1)                            # (order+1,)
        nfp_m_angle = nfp * m[:, None] * angle[None, :]     # (order+1, N)
        cos_nm = jnp.cos(nfp_m_angle)
        sin_nm = jnp.sin(nfp_m_angle)

        rc = dofs[: order + 1]
        r = jnp.sum(rc[:, None] * cos_nm, axis=0)

        if self.stellsym:
            zs = dofs[order + 1 :]
            z = jnp.sum(zs[:, None] * sin_nm[1:], axis=0)
        else:
            # [rc (order+1), rs (order), zc (order+1), zs (order)]
            rs = dofs[order + 1 : 2 * order + 1]
            zc = dofs[2 * order + 1 : 3 * order + 2]
            zs = dofs[3 * order + 2 :]
            r = r + jnp.sum(rs[:, None] * sin_nm[1:], axis=0)
            z = (
                jnp.sum(zc[:, None] * cos_nm, axis=0)
                + jnp.sum(zs[:, None] * sin_nm[1:], axis=0)
            )

        x = r * jnp.cos(angle)
        y = r * jnp.sin(angle)
        return jnp.stack([x, y, z], axis=-1).reshape(phi_shape + (3,))

    def gamma_eval(self, phi, diff_order: int = 0):
        """Evaluate the curve or its derivative at arbitrary parameter values.

        Positions use the cylindrical Fourier series; derivatives of order
        ``diff_order > 0`` are obtained by recursive ``jax.jvp``.

        Parameters
        ----------
        phi : array-like
            Curve parameter values in ``[0, 1)``, arbitrary shape.
        diff_order : int
            Derivative order (0 = positions, 1 = first derivative, …).

        Returns
        -------
        jax.Array, shape ``phi.shape + (3,)``
        """
        phi = jnp.asarray(phi, dtype=float)
        if diff_order == 0:
            return self._gamma_pos(phi)

        # Recursive JVP: d^n gamma / d phi^n
        def nth_deriv(order, p):
            if order == 0:
                return self._gamma_pos(p)
            _, tangent = jax.jvp(
                lambda q: nth_deriv(order - 1, q),
                (p,),
                (jnp.ones_like(p),),
            )
            return tangent

        return nth_deriv(diff_order, phi)


def curve_jax_from_simsopt(curve) -> CurveJAX:
    """Convert a simsopt curve (or pass through an existing :class:`CurveJAX`).

    Parameters
    ----------
    curve : CurveJAX or simsopt CurveXYZFourier / CurveRZFourier
        Input curve.

    Returns
    -------
    CurveJAX
    """
    if isinstance(curve, CurveJAX):
        return curve
    # Lazy import to avoid requiring simsopt at module load.
    from simsopt.geo import CurveXYZFourier, CurveRZFourier

    if isinstance(curve, CurveXYZFourier):
        return CurveXYZFourierJAX.from_simsopt(curve)
    if isinstance(curve, CurveRZFourier):
        return CurveRZFourierJAX.from_simsopt(curve)
    raise TypeError(
        f"Unsupported curve type {type(curve)!r}; expected CurveJAX, "
        "CurveXYZFourier, or CurveRZFourier."
    )
