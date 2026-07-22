"""Cross-section property factories for bisymmetric beam elements.

Every function in this module is compatible with the ``cross_section_fn``
attribute of :class:`~coil_fem.simsopt.CoilSupportBeams` ad 
:class:`~coil_fem.coupling.SupportBeams`:

.. code-block:: python

    A, Iy, Iz, J = fn(support_dofs)  # each shape (n_base, n_beams_per_coil)

The geometric parameters are read from ``support_dofs`` by key, so they can
live alongside the attachment-angle DOFs and be treated as optimizable
variables by simsopt. Each function must be accompanied by a ``func_name_keys``
to store the required keys in support_dofs.

Conventions (matching ``docs/theory/bisymbeam.rst``)
-----------------------------------------------------
Local beam frame: x along the centroidal axis, y and z as cross-section axes.

* ``A`` — the cross section of the beam.
* ``I_y`` — second moment of area about the local y-axis; governs bending in
  the x–z plane (deflection w).  For a rectangle this is ``h b³ / 12`` where b
  is the z-extent of the section.
* ``I_z`` — second moment of area about the local z-axis; governs bending in
  the x–y plane (deflection v).  For a rectangle this is ``b h³ / 12`` where h
  is the y-extent.
* ``J``  — St. Venant torsion constant.  Equals the polar moment
  ``I_y + I_z`` **only** for circular sections.
"""

from __future__ import annotations

import math
from typing import Callable

import jax.numpy as jnp

def bool_to_sign(a):
    if a:
        return 1
    else:
        return -1

# ============================================================================
# Solid circle
# ============================================================================

solid_circle_keys = ('r_beam',)

def solid_circle(support_dofs: dict):
    """Cross-section function for a solid circular section.

    Reads the beam radius from ``support_dofs['r_beam']``. 
    
    For a solid circle with radius :math:`r_beam`:

    .. math::

        A = \\pi r^2, \\quad
        I_y = I_z = \\frac{\\pi r^4}{4}, \\quad
        J = \\frac{\\pi r^4}{2}.

    :math:`J` equals the polar moment :math:`I_p = I_y + I_z` because of
    full circular symmetry.

    Parameters
    ----------
    support_dofs : dict
        Must contain the entry 'r_beam'.

    Returns
    -------
    ``(A, Iy, Iz, J)`` 
        Each output has the same shape as ``support_dofs[radius_key]``.

    """
    r = support_dofs['r_beam']
    A  = jnp.pi * r ** 2
    I  = jnp.pi * r ** 4 / 4.0
    J  = jnp.pi * r ** 4 / 2.0
    return A, I, I, J

def solid_circle_clamp(surface_pts_beam_frame, clamp_point, sign_x, dofs, constants):
    pts_x = surface_pts_beam_frame[:, 0]
    pts_y = surface_pts_beam_frame[:, 1]
    pts_z = surface_pts_beam_frame[:, 2]
    r = dofs['r_beam']
    
    in_correct_direction = jnp.where(
        bool_to_sign(sign_x) * pts_x <= 0,
        1
    )

    in_long_cylinder = jnp.where(
        pts_y**2 + pts_z**2 <= r**2,
        1
    )

# ============================================================================
# Solid rectangle
# ============================================================================


solid_rectangle_keys = ('w1_beam', 'w2_beam',)
    
def solid_rectangle(support_dofs : dict):
    """Cross-section function for a solid rectangular section.

    Reads the beam widths from ``support_dofs['w1_beam']`` and ``support_dofs['w2_beam']``. 
    
    .. math::

        A = w_1 w_2, \\quad
        I_y = \\frac{w_2\\,w_1^3}{12}, \\quad
        I_z = \\frac{w_1\\,w_2^3}{12}.

    The St. Venant torsion constant uses the Roark / Pilkey approximation
    (accurate to within 0.1% for any aspect ratio):

    .. math::

        J = a\\,c^3 \\left(\\frac{1}{3}
            - 0.21\\frac{c}{a}\\left(1 - \\frac{c^4}{12\\,a^4}\\right)\\right),

    where :math:`a = \\max(b, h)` and :math:`c = \\min(b, h)`.
    This reduces to :math:`\\tfrac{1}{3}ac^3` for thin strips (:math:`c\\ll a`).
    Parameters
    ----------
    support_dofs : dict
        Must contain the entry 'w1_beam' and 'w2_beam' the z and y widths :math:`w_1` and 
        :math:`w_2`.

    Returns
    -------
    ``(A, Iy, Iz, J)`` 
        Each output has the same shape as ``support_dofs[radius_key]``.

    
    """
    w1 = support_dofs['w1_beam']    # z-extent
    w2 = support_dofs['w2_beam']   # y-extent

    A  = w1 * w2
    Iy = w2 * w1 ** 3 / 12.0        # governs w (z-deflection)
    Iz = w1 * w2 ** 3 / 12.0        # governs v (y-deflection)

    # Roark/Pilkey torsion constant (JAX-differentiable)
    a_rect = jnp.maximum(w1, w2)
    c_rect = jnp.minimum(w1, w2)
    ratio  = c_rect / a_rect       # ≤ 1
    J = a_rect * c_rect ** 3 * (
        1.0 / 3.0
        - 0.21 * ratio * (1.0 - ratio ** 4 / 12.0)
    )
    return A, Iy, Iz, J


# ============================================================================
# Hollow circle (annular section)
# ============================================================================

hollow_circle_keys = ('r_1_beam', 'r_1_beam',)
    
def hollow_circle(support_dofs: dict) -> Callable:
    """Cross-section function for a hollow circular section.

    Reads the beam radii from ``support_dofs``. 
    
    For a hollow cylinder with two radii :math:`r_1` and :math:`r_2`.
    The larger will be chosen as the outer radius.

    .. math::

        A = \\pi(r_2^2 - r_1^2), \\quad
        I_y = I_z = \\frac{\\pi(r_2^4 - r_1^4)}{4}, \\quad
        J = \\frac{\\pi(r_2^4 - r_1^4)}{2}.

    *J* equals the polar moment :math:`I_p = I_y + I_z` because of circular
    symmetry.  No thin-wall approximation is made.

    Parameters
    ----------
    support_dofs : dict
        Must contain the entry 'r_1_beam' and 'r_2_beam'.

    Returns
    -------
    ``(A, Iy, Iz, J)`` 
        Each output has the same shape as ``support_dofs[radius_key]``.

    """
    r_o = jnp.maximum(support_dofs['r_1_beam'], support_dofs['r_2_beam'])
    r_i = jnp.minimum(support_dofs['r_1_beam'], support_dofs['r_2_beam'])

    A  = jnp.pi * (r_o ** 2 - r_i ** 2)
    I  = jnp.pi * (r_o ** 4 - r_i ** 4) / 4.0
    J  = jnp.pi * (r_o ** 4 - r_i ** 4) / 2.0
    return A, I, I, J