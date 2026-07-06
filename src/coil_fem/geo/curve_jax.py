"""
Pure JAX implementation of CurveXYZFourier as a JAX pytree.

DOF layout (per coordinate x, y, z)::

    [c0, s1, c1, s2, c2, ..., s_order, c_order]

where ``ci`` is the cosine coefficient for mode *i* and ``si`` the sine coefficient.
The full ``dofs`` array has shape ``(3*(2*order+1),)``.

Pytree leaves (traced): ``quadpoints``, ``dofs``. Pytree aux (static): ``order``.
"""

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class CurveXYZFourierJAX:
    r"""
    JAX pytree implementation of CurveXYZFourier.

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

    # ------------------------------------------------------------------ #
    # JAX pytree protocol                                                   #
    # ------------------------------------------------------------------ #

    def tree_flatten(self):
        children = (self.quadpoints, self.dofs)
        aux_data = self.order
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        quadpoints, dofs = children
        return cls(quadpoints, dofs, aux_data)

    # ------------------------------------------------------------------ #
    # Simsopt interop                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_simsopt(cls, curve):
        """
        Construct a CurveXYZFourierJAX from a simsopt CurveXYZFourier.

        The DOF layout is identical between the two representations
        ([c0, s1, c1, …] per coordinate), so no reordering is needed.
        """
        return cls(
            quadpoints=curve.quadpoints,
            dofs=curve.get_dofs(),
            order=curve.order,
        )

    def to_simsopt(self):
        """
        Convert this CurveXYZFourierJAX to a simsopt CurveXYZFourier.

        Returns a new simsopt CurveXYZFourier with the same quadpoints,
        order, and DOFs.
        """
        import numpy as np
        from simsopt.geo import CurveXYZFourier

        curve = CurveXYZFourier(np.asarray(self.quadpoints), self.order)
        curve.set_dofs(np.asarray(self.dofs))
        return curve

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def gamma_eval(self, phi, diff_order: int = 0):
        """
        Evaluate the curve (or its derivative) at arbitrary angles phi.

        Uses the analytic formula for the n-th derivative of a Fourier series:

            d^n/dphi^n sin(2*pi*j*phi) = (2*pi*j)^n * sin(2*pi*j*phi + n*pi/2)
            d^n/dphi^n cos(2*pi*j*phi) = (2*pi*j)^n * cos(2*pi*j*phi + n*pi/2)

        Args:
            phi:        Array-like of angles in [0, 1), arbitrary shape.
            diff_order: Non-negative integer; 0 returns positions, 1 returns
                        first derivatives, etc.

        Returns:
            Array of shape ``phi.shape + (3,)``.
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
        """
        Curvature kappa = ||gamma' x gamma''|| / ||gamma'||^3.

        Returns:
            Array of shape (nquad,).
        """
        d1 = self.gammadash()
        d2 = self.gammadashdash()
        return (
            jnp.linalg.norm(jnp.cross(d1, d2), axis=1)
            / jnp.linalg.norm(d1, axis=1) ** 3
        )

    def torsion(self):
        """
        Torsion tau = (gamma' x gamma'') . gamma''' / ||gamma' x gamma''||^2.

        Returns:
            Array of shape (nquad,).
        """
        d1 = self.gammadash()
        d2 = self.gammadashdash()
        d3 = self.gammadashdashdash()
        cross = jnp.cross(d1, d2)                        # (nquad, 3)
        return jnp.sum(cross * d3, axis=1) / jnp.sum(cross ** 2, axis=1)

    def incremental_arclength(self):
        """
        Incremental arclength at each quadrature point.

        Returns the differential arclength element:
            ds = ||gamma'(phi)|| dphi

        where gamma'(phi) is the derivative of the position vector with respect
        to the parameterization variable phi.

        Returns:
            Array of shape (nquad,) containing ||gamma'|| at each quadrature point.
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
