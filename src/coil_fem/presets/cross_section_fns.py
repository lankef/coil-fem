"""Cross-section property factories for bisymmetric beam elements.

Every function in this module is compatible with the ``cross_section_fn``
attribute of :class:`~coil_fem.simsopt.CoilSupportBeams` and
:class:`~coil_fem.coupling.SupportBeams`:

.. code-block:: python

    A, Iy, Iz, J = fn(support_dofs)  # each a per-group list of arrays

``support_dofs`` values are ragged: a Python list with one entry per CC/CF
group, each entry an array of per-beam values for that group (see
:class:`~coil_fem.coupling.SupportBeams`).  These lists are valid JAX
pytrees, so each function below maps its elementwise formula across groups
via ``jax.tree_util.tree_map`` (see ``_map_groups``) rather than operating
on a flat array directly.

The geometric parameters are read from ``support_dofs`` by key, so they can
live alongside the attachment-angle DOFs and be treated as optimizable
variables by simsopt. Each function must be accompanied by a ``func_name_dof_keys``
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

import jax
import jax.numpy as jnp

from ..utils import clamp_sigmoid


def _bool_to_sign(a):
    if a:
        return 1
    else:
        return -1


def _map_groups(fn, *ragged_args):
    """Apply a per-group cross-section formula across ragged support_dofs entries.

    ``support_dofs`` values are Python lists of per-group JAX arrays (one
    entry per CC/CF group), which are themselves valid JAX pytrees (the list
    is the container, each array is a leaf).  ``fn`` implements the
    elementwise ``(A, Iy, Iz, J)`` formula for a single group's array; this
    helper maps it across groups and transposes the per-group tuples back
    into the four per-group lists expected by ``SupportBeams.coo``.

    Parameters
    ----------
    fn : callable
        ``fn(*leaf_arrays) -> (A, Iy, Iz, J)`` for one group.
    *ragged_args : list of jax.Array
        One or more per-group lists (all with the same length).

    Returns
    -------
    tuple of list
        ``(A, Iy, Iz, J)``, each a per-group list of arrays.
    """
    results = jax.tree_util.tree_map(fn, *ragged_args)
    return tuple(list(x) for x in zip(*results))


wrap_option_keys = ('r_attachment', 'eps_sigmoid',)


def wrap_attachment(surface_pts_beam_frame, dofs, sign_x, beam_options):
    """Select surface points within a spherical radius of the beam endpoint.

    Uses a soft ball of radius ``beam_options['r_attachment']`` centred at the
    endpoint (origin of ``surface_pts_beam_frame``).  Independent of the beam
    cross-section shape; useful when the mesh is too coarse for the direct
    section-shaped attachment to capture torsion.

    Parameters
    ----------
    surface_pts_beam_frame : jax.Array, shape (N, 3)
        Surface points expressed in the beam's local frame, origin at the
        endpoint.
    dofs : dict
        Unused; accepted for API compatibility with section-shaped attachments.
    sign_x : bool
        Unused; accepted for API compatibility.
    beam_options : dict
        Must contain ``'r_attachment'`` and ``'eps_sigmoid'``.

    Returns
    -------
    jax.Array, shape (N,)
        Soft attachment weights in ``[0, 1]``.
    """
    pts_x = surface_pts_beam_frame[:, 0]
    pts_y = surface_pts_beam_frame[:, 1]
    pts_z = surface_pts_beam_frame[:, 2]

    d_sq = pts_x**2 + pts_y**2 + pts_z**2
    return clamp_sigmoid(
        d_sq=d_sq,
        r=beam_options['r_attachment'],
        eps_sigmoid=beam_options['eps_sigmoid'],
    )


# ============================================================================
# Solid circle
# ============================================================================

solid_circle_dof_keys = ('r_beam',)
solid_circle_option_keys = ('eps_sigmoid',)


def solid_circle(support_dofs: dict):
    """Cross-section function for a solid circular section.

    Reads the beam radius from ``support_dofs['r_beam']``.

    For a solid circle with radius :math:`r`:

    .. math::

        A = \\pi r^2, \\quad
        I_y = I_z = \\frac{\\pi r^4}{4}, \\quad
        J = \\frac{\\pi r^4}{2}.

    :math:`J` equals the polar moment :math:`I_p = I_y + I_z` because of
    full circular symmetry.

    Parameters
    ----------
    support_dofs : dict
        Must contain the entry ``'r_beam'``.

    Returns
    -------
    ``(A, Iy, Iz, J)``
        Each output is a per-group list matching the structure of
        ``support_dofs['r_beam']``.
    """
    def _single(r):
        A = jnp.pi * r ** 2
        I = jnp.pi * r ** 4 / 4.0
        J = jnp.pi * r ** 4 / 2.0
        return A, I, I, J

    return _map_groups(_single, support_dofs['r_beam'])


def solid_circle_attachment(surface_pts_beam_frame, dofs, sign_x, beam_options):
    """Select surface points inside the solid circular beam cross-section.

    Soft indicator of the disk of radius ``dofs['r_beam']`` in the local
    y–z plane, gated to the half-space of the beam (``sign_x``).

    Parameters
    ----------
    surface_pts_beam_frame : jax.Array, shape (N, 3)
        Surface points expressed in the beam's local frame, origin at the
        endpoint.
    dofs : dict
        Must contain ``'r_beam'`` (scalar for this beam).
    sign_x : bool
        ``True`` at the node-1 end (beam extends toward ``+x_local``);
        ``False`` at node-2.
    beam_options : dict
        Must contain ``'eps_sigmoid'``.

    Returns
    -------
    jax.Array, shape (N,)
        Soft attachment weights in ``[0, 1]``.
    """
    pts_x = surface_pts_beam_frame[:, 0]
    pts_y = surface_pts_beam_frame[:, 1]
    pts_z = surface_pts_beam_frame[:, 2]

    in_correct_direction = jnp.where(
        _bool_to_sign(sign_x) * pts_x >= 0,
        1, 0
    )
    d_sq = pts_y**2 + pts_z**2
    return in_correct_direction * clamp_sigmoid(
        d_sq=d_sq,
        r=dofs['r_beam'],
        eps_sigmoid=beam_options['eps_sigmoid'],
    )


# ============================================================================
# Solid rectangle
# ============================================================================

solid_rectangle_dof_keys = ('w1_beam', 'w2_beam',)
solid_rectangle_option_keys = ('eps_sigmoid',)


def solid_rectangle(support_dofs: dict):
    """Cross-section function for a solid rectangular section.

    Reads the beam widths from ``support_dofs['w1_beam']`` (z-extent) and
    ``support_dofs['w2_beam']`` (y-extent).

    .. math::

        A = w_1 w_2, \\quad
        I_y = \\frac{w_2\\,w_1^3}{12}, \\quad
        I_z = \\frac{w_1\\,w_2^3}{12}.

    The St. Venant torsion constant uses the Roark / Pilkey approximation
    (accurate to within 0.1% for any aspect ratio):

    .. math::

        J = a\\,c^3 \\left(\\frac{1}{3}
            - 0.21\\frac{c}{a}\\left(1 - \\frac{c^4}{12\\,a^4}\\right)\\right),

    where :math:`a = \\max(w_1, w_2)` and :math:`c = \\min(w_1, w_2)`.
    This reduces to :math:`\\tfrac{1}{3}ac^3` for thin strips (:math:`c\\ll a`).

    Parameters
    ----------
    support_dofs : dict
        Must contain ``'w1_beam'`` and ``'w2_beam'``, the z and y widths
        :math:`w_1` and :math:`w_2`.

    Returns
    -------
    ``(A, Iy, Iz, J)``
        Each output is a per-group list matching the structure of
        ``support_dofs['w1_beam']`` / ``support_dofs['w2_beam']``.
    """
    def _single(w1, w2):
        A = w1 * w2
        Iy = w2 * w1 ** 3 / 12.0        # governs w (z-deflection)
        Iz = w1 * w2 ** 3 / 12.0        # governs v (y-deflection)

        # Roark/Pilkey torsion constant (JAX-differentiable)
        a_rect = jnp.maximum(w1, w2)
        c_rect = jnp.minimum(w1, w2)
        ratio = c_rect / a_rect       # ≤ 1
        J = a_rect * c_rect ** 3 * (
            1.0 / 3.0
            - 0.21 * ratio * (1.0 - ratio ** 4 / 12.0)
        )
        return A, Iy, Iz, J

    return _map_groups(
        _single, support_dofs['w1_beam'], support_dofs['w2_beam'],
    )


def solid_rectangle_attachment(surface_pts_beam_frame, dofs, sign_x, beam_options):
    """Select surface points inside the solid rectangular beam cross-section.

    Soft indicator of the rectangle of half-widths
    ``dofs['w1_beam']/2`` (z) and ``dofs['w2_beam']/2`` (y), gated to the
    half-space of the beam (``sign_x``).

    Parameters
    ----------
    surface_pts_beam_frame : jax.Array, shape (N, 3)
        Surface points expressed in the beam's local frame, origin at the
        endpoint.
    dofs : dict
        Must contain ``'w1_beam'`` and ``'w2_beam'`` (scalars for this beam).
    sign_x : bool
        ``True`` at the node-1 end (beam extends toward ``+x_local``);
        ``False`` at node-2.
    beam_options : dict
        Must contain ``'eps_sigmoid'``.

    Returns
    -------
    jax.Array, shape (N,)
        Soft attachment weights in ``[0, 1]``.
    """
    pts_x = surface_pts_beam_frame[:, 0]
    pts_y = surface_pts_beam_frame[:, 1]
    pts_z = surface_pts_beam_frame[:, 2]

    in_correct_direction = jnp.where(
        _bool_to_sign(sign_x) * pts_x >= 0,
        1, 0
    )
    eps = beam_options['eps_sigmoid']
    return in_correct_direction * clamp_sigmoid(
        d_sq=pts_z ** 2, r=dofs['w1_beam'] / 2, eps_sigmoid=eps,
    ) * clamp_sigmoid(
        d_sq=pts_y ** 2, r=dofs['w2_beam'] / 2, eps_sigmoid=eps,
    )


# ============================================================================
# Hollow circle (annular section)
# ============================================================================

hollow_circle_dof_keys = ('r_1_beam', 'r_2_beam',)
hollow_circle_option_keys = ('eps_sigmoid',)


def hollow_circle(support_dofs: dict):
    """Cross-section function for a hollow circular section.

    Reads the beam radii from ``support_dofs``.

    For a hollow cylinder with two radii :math:`r_1` and :math:`r_2`.
    The larger will be chosen as the outer radius.

    .. math::

        A = \\pi(r_o^2 - r_i^2), \\quad
        I_y = I_z = \\frac{\\pi(r_o^4 - r_i^4)}{4}, \\quad
        J = \\frac{\\pi(r_o^4 - r_i^4)}{2}.

    *J* equals the polar moment :math:`I_p = I_y + I_z` because of circular
    symmetry.  No thin-wall approximation is made.

    Parameters
    ----------
    support_dofs : dict
        Must contain the entry ``'r_1_beam'`` and ``'r_2_beam'``.

    Returns
    -------
    ``(A, Iy, Iz, J)``
        Each output is a per-group list matching the structure of
        ``support_dofs['r_1_beam']`` / ``support_dofs['r_2_beam']``.
    """
    def _single(r1, r2):
        r_o = jnp.maximum(r1, r2)
        r_i = jnp.minimum(r1, r2)
        A = jnp.pi * (r_o ** 2 - r_i ** 2)
        I = jnp.pi * (r_o ** 4 - r_i ** 4) / 4.0
        J = jnp.pi * (r_o ** 4 - r_i ** 4) / 2.0
        return A, I, I, J

    return _map_groups(
        _single, support_dofs['r_1_beam'], support_dofs['r_2_beam'],
    )


hollow_circle_attachment = solid_circle_attachment


# ============================================================================
# Hollow rectangle (uniform wall thickness)
# ============================================================================

hollow_rectangle_dof_keys = ('w1_beam', 'w2_beam', 't_beam',)
hollow_rectangle_option_keys = ('eps_sigmoid',)


def hollow_rectangle(support_dofs: dict):
    """Cross-section function for a hollow rectangular section of uniform thickness.

    Reads the outer widths from ``support_dofs['w1_beam']`` (z-extent) and
    ``support_dofs['w2_beam']`` (y-extent), and the wall thickness from
    ``support_dofs['t_beam']``.  Thickness is clipped to
    :math:`\\min(w_1, w_2)/2` so an oversized value degrades to the solid
    rectangle rather than a negative area.

    With inner dimensions :math:`b_i = w_1 - 2t` and :math:`h_i = w_2 - 2t`:

    .. math::

        A = w_1 w_2 - b_i h_i, \\quad
        I_y = \\frac{w_2 w_1^3 - h_i b_i^3}{12}, \\quad
        I_z = \\frac{w_1 w_2^3 - b_i h_i^3}{12}.

    The St. Venant torsion constant uses Bredt's closed thin-wall formula
    (Roark Table 10.1, case 16, specialised to uniform :math:`t`):

    .. math::

        J = \\frac{2 t (w_1 - t)^2 (w_2 - t)^2}{w_1 + w_2 - 2 t},

    equivalent to :math:`4 A_m^2 / \\oint (ds / t)` with midline area
    :math:`A_m = (w_1 - t)(w_2 - t)`.

    Notes
    -----
    The Bredt formula is a thin-wall approximation and is accurate when
    :math:`t \\ll \\min(w_1, w_2)`.  Unlike circular sections,
    :math:`J \\neq I_y + I_z`.

    Parameters
    ----------
    support_dofs : dict
        Must contain ``'w1_beam'``, ``'w2_beam'``, and ``'t_beam'``.

    Returns
    -------
    ``(A, Iy, Iz, J)``
        Each output is a per-group list matching the structure of
        ``support_dofs['w1_beam']`` / ``support_dofs['w2_beam']`` /
        ``support_dofs['t_beam']``.
    """
    def _single(w1, w2, t):
        t = jnp.minimum(t, jnp.minimum(w1, w2) / 2.0)
        b_i = w1 - 2.0 * t
        h_i = w2 - 2.0 * t
        A = w1 * w2 - b_i * h_i
        Iy = (w2 * w1 ** 3 - h_i * b_i ** 3) / 12.0
        Iz = (w1 * w2 ** 3 - b_i * h_i ** 3) / 12.0
        # Bredt: J = 4 A_m^2 / ∮(ds/t), A_m = (w1-t)(w2-t), uniform t
        J = 2.0 * t * (w1 - t) ** 2 * (w2 - t) ** 2 / (w1 + w2 - 2.0 * t)
        return A, Iy, Iz, J

    return _map_groups(
        _single,
        support_dofs['w1_beam'],
        support_dofs['w2_beam'],
        support_dofs['t_beam'],
    )


hollow_rectangle_attachment = solid_rectangle_attachment
