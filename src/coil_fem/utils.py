"""General-purpose math helpers shared across the package.

Small, self-contained numerical formulas (clamp weighting, interpolation)
that don't belong to any single physics module.
"""

from jax.nn import sigmoid


def clamp_sigmoid(d_sq, r, eps_sigmoid):
    sigmoid_width = eps_sigmoid * r
    return sigmoid((r**2 - d_sq) / (sigmoid_width**2))


def cubic_hermite_interp(xi, L, y0, dy0, y1, dy1):
    """Cubic Hermite interpolation between two endpoints with prescribed slopes.

    Evaluates ``y(xi)`` for ``xi`` in ``[0, 1]`` given endpoint values ``y0``,
    ``y1`` and *physical* slopes ``dy0 = dy/dx|_{x=0}``, ``dy1 = dy/dx|_{x=L}``
    (not normalized w.r.t. ``xi``). This is the exact homogeneous solution of
    ``y'''' = 0`` matching four boundary conditions, e.g. Euler-Bernoulli beam
    bending between two nodes with no distributed load.

    Parameters
    ----------
    xi : jax.Array or float
        Normalized position(s) in [0, 1].
    L : jax.Array or float
        Physical span between the two endpoints.
    y0, y1 : jax.Array or float
        Endpoint values.
    dy0, dy1 : jax.Array or float
        Endpoint slopes ``dy/dx`` (physical, not w.r.t. ``xi``).

    Returns
    -------
    jax.Array
        ``y(xi)``, broadcast shape of the inputs.
    """
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L * (xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L * (xi**3 - xi**2)
    return N1*y0 + N2*dy0 + N3*y1 + N4*dy1


def estimate_k(L, E, eps):
    r"""Estimate the boundary spring coefficient based on scalelengths and properties.

    This estimate of is based on the following formula:
    
    .. math::
        \frac{FL^3}{3EI} \approx \epsilon^{-1} \left(\frac{12FL^2}{kw^4}\right)

    Here, the LHS is the displacement of a cantilever of length L due to end force F.
    The RHS is the displacement of a rigid beam attached to a square area with area
    spring density k:
     
    .. math::
        FL = k \delta \frac{w}{L}\frac{w^3}{12}  
    
    Physically, this means that the displacement of a cantilever's end 
    due to its spring foundation must be much smaller than the displacement due to its
    bending.

    Parameters
    ----------
    L : jax.Array or float
        The length of a beam.
    E : jax.Array or float
        The Young's modulus.
    eps : jax.Array or float
        A small factor. Recommended to be 1e-3 to 1e-4.

    Returns
    -------
    float
        Estimated k [N/m³], suitable for :class:`~coil_fem.coupling.Support` 
        as ``k_clamp``.
    """
    return E/L/eps
