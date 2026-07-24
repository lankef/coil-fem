"""Beam-network support structure.

Provides :class:`SupportBeams`, which models a cage-type support structure
with two types of support beams: 

1. Coil-coil (CC) beams that link adjacent coils.
2. Coil-foundation (CF) that link a coil to the foundation (fixed points in space).

All beams are treated as bisymmetric frame elements (McGuire, Gallagher & Ziemian, Eq. 4.34;
see ``docs/theory/bisymbeam.rst``).  Each beam endpoint couples to coil
exterior mesh points via translational and torsional springs whose spatial
distribution is governed by a user-supplied ``attachment_fn``.

:meth:`SupportBeams.coo` returns the support-local stiffness block
``K_ss`` in COO format (differentiable w.r.t. all traced inputs).
:meth:`SupportBeams.solve` runs a standalone lineax forward solve for the
beam DOFs given coil-side mesh displacements.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import lineax

from .supports import Support
from ..geo.curve_jax import CurveXYZFourierJAX


# ============================================================================
# Internal beam-frame helpers (module-level, pure jnp)
# ============================================================================

def _rodrigues(axis: jax.Array, angle: jax.Array) -> jax.Array:
    """Rotation matrix rotating by ``angle`` (radians) about unit ``axis``.

    Parameters
    ----------
    axis : jax.Array, shape (3,)
        Unit rotation axis.
    angle : jax.Array, scalar
        Rotation angle in radians.

    Returns
    -------
    jax.Array, shape (3, 3)
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    x, y, z = axis[0], axis[1], axis[2]
    # Rodrigues formula: R = c*I + (1-c)*outer(axis,axis) + s*skew(axis)
    outer = jnp.outer(axis, axis)
    skew = jnp.array([[ 0., -z,  y],
                      [ z,  0., -x],
                      [-y,  x,  0.]])
    return c * jnp.eye(3) + (1.0 - c) * outer + s * skew


def _skew(v: jax.Array) -> jax.Array:
    """3×3 skew-symmetric (cross-product) matrix for vector ``v``.

    Parameters
    ----------
    v : jax.Array, shape (3,)

    Returns
    -------
    jax.Array, shape (3, 3)
        ``[v]×`` such that ``[v]× w = v × w``.
    """
    return jnp.array([[ 0.,    -v[2],  v[1]],
                      [ v[2],  0.,    -v[0]],
                      [-v[1],  v[0],   0.  ]])


def _gamma_12(gamma_3: jax.Array) -> jax.Array:
    """Expand a 3×3 direction-cosine matrix into the 12×12 beam DCM.

    The 12-DOF transformation matrix is block-diagonal with four copies of
    ``gamma_3`` (one block per DOF triad: node-1 translations, node-1
    rotations, node-2 translations, node-2 rotations).

    Parameters
    ----------
    gamma_3 : jax.Array, shape (3, 3)

    Returns
    -------
    jax.Array, shape (12, 12)
    """
    z = jnp.zeros((3, 3))
    row0 = jnp.concatenate([gamma_3, z, z, z], axis=1)
    row1 = jnp.concatenate([z, gamma_3, z, z], axis=1)
    row2 = jnp.concatenate([z, z, gamma_3, z], axis=1)
    row3 = jnp.concatenate([z, z, z, gamma_3], axis=1)
    return jnp.concatenate([row0, row1, row2, row3], axis=0)


# ============================================================================
# SupportBeams
# ============================================================================

class SupportBeams(Support):
    """Bisymmetric beam-network support.

    Models connections between adjacent coils (coil-coil, CC) and between
    coils and fixed anchor points (coil-foundation, CF) using independent
    bisymmetric space-frame elements.  Inherits the grounded Winkler weight
    machinery from :class:`Support` when ``fixed_clamp_fns`` is provided.

    Parameters
    ----------
    n_base : int
        Number of base coils (before symmetry expansion).
    beam_options : dict
        Contains the following entries
        n_beam_cc : int or sequence of int
            Number of coil-coil beams per CC group.  A scalar is broadcast to
            every group; a sequence must have one entry per group:
            ``n_base + 1`` entries when ``stellsym=True`` (the extra last
            entry is the coil-0 ``phi = 0`` wrap group), else ``n_base``.
            See the *CC beam groups* note below.
        n_beam_cf : int or sequence of int
            Number of coil-foundation beams per base coil.  Scalar (broadcast)
            or length-``n_base`` sequence.
        E : float
            Young's modulus [Pa].
        nu : float
            Poisson ratio.
        k_lin : float
            Translational spring stiffness [N/m²] (applied as ``k_lin * w``
            per surface node).
        k_tor : float
            Torsional spring stiffness [N·m/m²] (applied as ``k_tor * w``
            per surface node).
        and additional constants that ``attachment_fn`` needs.
    cross_section_fn : callable
        ``cross_section_fn(support_dofs) -> (A, Iy, Iz, J)`` where each
        returned value is a per-group list of arrays: entry ``i < n_base``
        has shape ``(n_beam_cc[i] + n_beam_cf[i],)``; when ``stellsym=True``
        an extra entry ``n_base`` has shape ``(n_beam_cc[n_base],)`` (wrap
        group, no CF part).  Section-property validation is delegated to
        this callable.
    attachment_fn : callable
        ``attachment_fn(surface_pts_beam_frame, dofs, sign_x, beam_options) -> weights``
        where ``surface_pts_beam_frame`` is ``(n_surface_nodes, 3)`` — surface
        points expressed in the beam's local frame with origin at the endpoint
        (computed as ``(pts − x_endpoint) @ Gamma_3``), ``dofs`` is the merged
        support-dofs dict, ``sign_x`` is ``True`` at the node-1 end (beam
        extends toward ``+x_local``) and ``False`` at node-2, and ``beam_options``
        is the options dict.  ``weights`` is ``(n_surface_nodes,)`` in
        ``[0, 1]``.
    fixed_clamp_fns : callable or list[callable] or None
        Optional Winkler weight functions forwarded to :class:`Support`.
        When set, :meth:`compute_weights` behaves identically to
        :class:`Support` regardless of the beam network.

    Notes
    -----
    **Beam DOF layout** (support-local numbering, one beam = 12 consecutive
    DOFs):

    * Beams are ordered coil-major, cc-then-cf.  With per-coil counts, coil
      ``i`` starts at the cumulative offset ``beam_offsets[i] =
      Σ_{i'<i} (n_beam_cc[i'] + n_beam_cf[i'])``; its CC beams occupy local
      slots ``0 … n_beam_cc[i]-1`` and its CF beams
      ``n_beam_cc[i] … n_beam_cc[i]+n_beam_cf[i]-1``.  When ``stellsym=True``
      the wrap group's beams (group ``n_base``) are appended after all
      per-coil blocks, starting at ``wrap_beam_offset``.
    * Local DOF ordering per beam follows Figure 4.6 of
      ``docs/theory/bisymbeam.rst``:
      ``[u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]``.
    * Beam local frame (see :meth:`_direction_cosine_matrices`):
      ``x_local`` is the beam axis (unit vector node1 → node2);
      ``z_local`` is ``cross(x_local, t_coil_start)`` — the direction normal
      to both the beam axis and the coil's tangent at the node-1 attachment
      point — rolled about ``x_local`` by ``thetas_orientation``;
      ``y_local = cross(z_local, x_local)`` completes the right-handed triad.
      ``z_local`` is therefore geometry-dependent (and optimizable via the
      roll angle), not a fixed global axis.

    **CC beam groups and symmetry wraparound**:

    CC beams are organised in ``n_groups_cc`` groups (``n_base + 1`` when
    ``stellsym=True``, else ``n_base``):

    * group ``i < n_base - 1``: connects ``base_coil[i]`` to
      ``base_coil[i+1]`` (no transform).
    * group ``n_base - 1``, ``stellsym=True``: connects the last coil to its
      stellarator reflection about the ``phi = pi/nfp`` half-period plane
      (``'flip_half'``).
    * group ``n_base - 1``, ``stellsym=False``: connects the last coil to
      the next field-period rotation of ``base_coil[0]`` (``'rotate'``).
    * group ``n_base`` (``stellsym=True`` only): connects ``base_coil[0]``
      to its stellarator reflection about the ``phi = 0`` plane (``'flip'``).

    Each boundary beam has a stellarator-symmetric partner beam whose
    attachment angles are the master's swapped
    (``phi_start <-> phi_end``).  Partner beams are never assembled:
    symmetry implies ``u_partner = Q u_master``, so the pair is represented
    exactly by the master's 12 DOFs plus ``Q``-transformed coupling blocks.
    All transforms ``Q`` are proper rotations (the stellarator flip is a C2
    rotation about x), so displacements, forces, torques and rotation DOFs
    transform alike.

    **Stiffness matrix symmetry**: the torque law
    ``τ = k_tor Σ w r × (u_attach − u_mesh)`` uses ``k_tor`` while the force
    law uses ``k_lin``, so the torque rows are not the transpose of the
    translation rows and ``K_ss`` is **not symmetric** in general.
    :meth:`solve` uses ``lineax`` which imposes no symmetry assumption.
    A future monolithic cuDSS path must use ``mtype_id=0`` (general).
    """

    def __init__(
        self,
        nfp: int,
        stellsym: bool,
        beam_options: dict,
        n_base: int,
        cross_section_fn: Callable,
        attachment_fn: Callable,
        cross_section_dof_keys: tuple = (),
        fixed_clamp_fns=None,
    ):
        super().__init__(fixed_clamp_fns=fixed_clamp_fns)

        self._n_base = int(n_base)
        self._beam_options = beam_options
        self._nfp = nfp
        self._stellsym = stellsym
        n_base = self._n_base
        # CC beams are organised in n_groups_cc groups: group i < n_base
        # connects coil i toward +phi (coil i+1, with a symmetry wrap at
        # i = n_base-1).  With stellsym an extra group n_base connects coil 0
        # to its phi = 0 stellarator image.
        self._n_groups_cc = n_base + (1 if stellsym else 0)
        # Beam counts: accept an int (broadcast) or a length-n_groups_cc
        # (cc) / length-n_base (cf) list/array.  Stored as tuples of ints.
        self._n_beam_cc = self._check_beam_counts(
            beam_options['n_beam_cc'], self._n_groups_cc, 'n_beam_cc'
        )
        # When stellsym==True, the beams connecting the last coil to its 
        # reflection (_n_beam_cc[-2]) and the first coil to its reflection 
        # (_n_beam_cc[-1]) are copied by reflection as well. Therefore, 
        # we HALF the corresponding _n_beam_cc entries to make sure that 
        # the final beam counts is still the even integer closest to the 
        # specified n_beam_cc.
        if stellsym:
            _n_beam_cc_temp = list(self._n_beam_cc)
            _n_beam_cc_temp[-1] = int(math.ceil(_n_beam_cc_temp[-1]/2))
            _n_beam_cc_temp[-2] = int(math.ceil(_n_beam_cc_temp[-2]/2))
            self._n_beam_cc = tuple(_n_beam_cc_temp)
        self._n_beam_cf = self._check_beam_counts(
            beam_options['n_beam_cf'], n_base, 'n_beam_cf'
        )
        # Cumulative global-beam offset per coil (coil-major, cc-then-cf order):
        # beam b of coil i starts at 12 * (_beam_offsets[i] + local_index).
        # The stellsym wrap group (group n_base) is appended after all
        # per-coil blocks, starting at _wrap_beam_offset.
        _per_coil = [self._n_beam_cc[i] + self._n_beam_cf[i] for i in range(n_base)]
        self._beam_offsets = tuple(int(sum(_per_coil[:i])) for i in range(n_base))
        self._wrap_beam_offset = int(sum(_per_coil))
        _n_wrap = self._n_beam_cc[n_base] if stellsym else 0
        self._n_beams_total = self._wrap_beam_offset + _n_wrap
        self._k_lin = float(beam_options['k_lin'])
        self._k_tor = float(beam_options['k_tor'])

        self.cross_section_fn = cross_section_fn
        self.attachment_fn = attachment_fn
        # Keys in support_dofs holding per-group lists of per-beam scalars;
        # _clamp_weights_for_spec slices [group, local_beam] before passing to
        # attachment_fn so each call receives a scalar, not the full array.
        self._cross_section_dof_keys = tuple(cross_section_dof_keys)

        # Wraparound symmetry transforms.  Every transform is a *proper*
        # rotation (det = +1): the stellarator flip diag(1,-1,-1) is a C2
        # rotation about x, so displacements, forces, torques and rotation
        # DOFs all transform by the same 3x3 matrix Q (no pseudovector
        # sign cases).  'flip'/'flip_half' are involutions (Q @ Q = I).
        _flip_Q = np.diag([1.0, -1.0, -1.0])
        _c = math.cos(2.0 * math.pi / nfp)
        _s = math.sin(2.0 * math.pi / nfp)
        _rot_Q = np.array([[_c, -_s, 0.0],
                           [_s,  _c, 0.0],
                           [0.0, 0.0, 1.0]])
        self._tfm_Q = {
            'none':      np.eye(3),
            'flip':      _flip_Q,            # stellsym about phi = 0
            'flip_half': _rot_Q @ _flip_Q,   # stellsym about phi = pi/nfp
            'rotate':    _rot_Q,             # next field period
        }

        # CC group topology: (start_coil, end_coil, transform_tag) per group.
        self._cc_groups = self._build_cc_groups()

        # Pre-build the static COO index arrays (I and J are loop-invariant).
        self._coo_I, self._coo_J = self._build_static_ij()

    # ============================================================================
    # Static topology helpers (called once at construction)
    # ============================================================================

    @staticmethod
    def _check_beam_counts(value, n_entries: int, name: str) -> tuple:
        """Broadcast/length-check a beam count integer/tuple. 
        
        This function broadcasts a beam count if it is an integer. If 
        it is a tuple/list, this function makes sure that contains
        ``n_entries`` entries.

        Parameters
        ----------
        value : int or sequence of int
            A scalar (broadcast to every entry) or a length-``n_entries``
            sequence.
        n_entries : int
            Number of entries (``n_base`` for CF beams; the number of CC
            groups — ``n_base + 1`` when ``stellsym=True`` — for CC beams).
        name : str
            Option name, used only in error messages.

        Returns
        -------
        tuple of int, length ``n_entries``
        """
        if np.ndim(value) == 0:
            return tuple(int(value) for _ in range(n_entries))
        seq = list(value)
        if len(seq) != n_entries:
            note = (
                " (n_beam_cc has one entry per CC group: n_base + 1 when "
                "stellsym=True — the extra last entry is the coil-0 phi=0 "
                "wrap group — else n_base)"
            ) if name == 'n_beam_cc' else ""
            raise ValueError(
                f"beam_options['{name}'] must be an int or a "
                f"length-{n_entries} sequence; got length {len(seq)}{note}."
            )
        return tuple(int(v) for v in seq)

    def _build_cc_groups(self) -> tuple:
        """Build the CC-beam group topology.

        Returns a tuple of ``(start_coil, end_coil, transform_tag)`` triples,
        one per CC group:

        * groups ``i < n_base - 1``: ``(i, i+1, 'none')``.
        * group ``n_base - 1``: last coil wraps — ``(n-1, n-1, 'flip_half')``
          when ``stellsym`` (reflection about ``phi = pi/nfp``), else
          ``(n-1, 0, 'rotate')`` (next field period).
        * group ``n_base`` (``stellsym`` only): ``(0, 0, 'flip')`` — coil 0
          to its ``phi = 0`` stellarator image.
        """
        groups = [(i, i + 1, 'none') for i in range(self.n_base - 1)]
        if self.stellsym:
            groups.append((self.n_base - 1, self.n_base - 1, 'flip_half'))
            groups.append((0, 0, 'flip'))
        else:
            groups.append((self.n_base - 1, 0, 'rotate'))
        return tuple(groups)

    def _build_static_ij(self):
        """Build flat COO row/column index arrays (static — no traced values).

        Returns
        -------
        I, J : np.ndarray, shape ``(n_beams_total * 144,)``
            Each beam contributes a 12×12 block on the block-diagonal.
        """
        n = self.n_beams_total
        # For beam b, block starts at global DOF offset = 12*b.
        # Entry (i_local, j_local) in that block maps to global (12*b+i, 12*b+j).
        b_idx = np.arange(n)                                   # (n,)
        local_i = np.arange(12)                                # (12,)
        local_j = np.arange(12)                                # (12,)
        # Outer: b x i x j
        I = (12 * b_idx[:, None, None]
             + local_i[None, :, None] * np.ones((n, 12, 12), dtype=int)).reshape(-1)
        J = (12 * b_idx[:, None, None]
             + local_j[None, None, :] * np.ones((n, 12, 12), dtype=int)).reshape(-1)
        return I.astype(np.int32), J.astype(np.int32)


    @property
    def is_coupled(self) -> bool:
        """``True`` — beams have their own DOFs coupled to coil surface nodes."""
        return True
    
    @property
    def nfp(self) -> bool:
        """``True`` — beams have their own DOFs coupled to coil surface nodes."""
        return self._nfp
        
    @property
    def beam_options(self) -> bool:
        """``True`` — beams have their own DOFs coupled to coil surface nodes."""
        return self._beam_options

    @property
    def stellsym(self) -> bool:
        """``True`` — beams have their own DOFs coupled to coil surface nodes."""
        return self._stellsym

    @property
    def n_beam_cc(self):
        """Per-group coil-coil beam counts, length-``n_groups_cc`` tuple of int."""
        return self._n_beam_cc 
        
    @property
    def n_beam_cf(self):
        """Per-coil coil-foundation beam counts, length-``n_base`` tuple of int."""
        return self._n_beam_cf

    @property
    def n_groups_cc(self):
        """Number of CC beam groups: ``n_base + 1`` when stellsym else ``n_base``."""
        return self._n_groups_cc

    @property
    def cc_groups(self):
        """CC group topology: tuple of ``(start_coil, end_coil, transform)``."""
        return self._cc_groups

    @property
    def beam_offsets(self):
        """Cumulative global-beam offset per coil, length-``n_base`` tuple of int."""
        return self._beam_offsets

    @property
    def wrap_beam_offset(self):
        """Global beam index where the stellsym wrap group (group ``n_base``) starts."""
        return self._wrap_beam_offset

    @property
    def k_lin(self):
        return self._k_lin 
        
    @property
    def k_tor(self):
        return self._k_tor 

    @property
    def n_base(self):
        return self._n_base
        
    @property
    def n_beams_per_coil(self):
        """Per-coil total beam counts (cc group i + cf coil i), length-``n_base``.

        Excludes the stellsym wrap group (group ``n_base``), whose beams are
        appended after all per-coil blocks.
        """
        return tuple(
            self._n_beam_cc[i] + self._n_beam_cf[i] for i in range(self.n_base)
        )
        
    @property
    def n_beams_total(self):
        return self._n_beams_total
        
    @property
    def n_support_dofs(self):
        """Each beam has 2 nodes, and each node carries 6 DOFs (linear system dofs, not optimizable dofs).
        The dofs are:
        3 translational: u (axial), v (bending in plane 1), w (bending in plane 2)
        3 rotational: θx (torsion), θy, θz (bending rotations)
        """
        return 12 * self.n_beams_total

    def displacement_at(self, state: dict, points: jax.Array) -> jax.Array:
        """Return beam-side displacement at query points.

        Parameters
        ----------
        state : dict
            State returned by :meth:`solve` (contains ``'u_s'``).
        points : jax.Array, shape ``(N, 3)``
            Query coordinates.

        Returns
        -------
        jax.Array, shape ``(N, 3)``

        Notes
        -----
        TODO(driver-integration): This is a placeholder that returns zeros.
        A proper Hermite-shape-function interpolation along beam elements
        should be added when the staggered/monolithic driver is implemented.
        See ``docs/developers/support_structure.rst`` for guidance.
        """
        return jnp.zeros((points.shape[0], 3), dtype=points.dtype)

    # ============================================================================
    # Geometry helpers
    # ============================================================================

    def _apply_end_transform(
        self, pts: jax.Array, transform: str, inverse: bool = False
    ) -> jax.Array:
        """Apply (or invert) the wraparound symmetry transform.

        Every transform is a proper rotation ``Q`` (see ``_tfm_Q``), so the
        same map applies to points, displacements, forces, torques and
        rotation DOFs alike.

        Parameters
        ----------
        pts : jax.Array, shape ``(..., 3)``
        transform : str
            One of ``'none'``, ``'flip'``, ``'flip_half'``, ``'rotate'``.
        inverse : bool
            When ``True``, apply ``Q^T`` (the inverse rotation).

        Returns
        -------
        jax.Array, same shape as ``pts``
        """
        Q = self._tfm_Q[transform]
        if inverse:
            return pts @ Q          # row-vector form of Q^T @ p
        return pts @ Q.T            # row-vector form of Q @ p

    def _beam_geometry(
        self,
        curves: list[CurveXYZFourierJAX],
        support_dofs: dict,
    ) -> dict:
        """Compute per-beam rest-state geometry arrays.

        All outputs are traced through ``curves`` and ``support_dofs`` and
        are therefore differentiable w.r.t. coil DOFs,
        ``phis_start_cc``, ``phis_end_cc``, ``phis_start_cf``,
        ``x_foundation``, etc.

        Parameters
        ----------
        curves : list[CurveXYZFourierJAX]
            Traced base-coil curve objects.
        support_dofs : dict
            Must contain ``phis_start_cc``, ``phis_end_cc``, ``phis_start_cf``,
            ``x_foundation`` (as specified in the class docstring).

        Returns
        -------
        dict with keys:

        * ``'x_start'``     : (N_beams, 3) — endpoint at node 1 (coil side, always).
        * ``'x_end'``       : (N_beams, 3) — endpoint at node 2.
        * ``'t_beam'``      : (N_beams, 3) — unit beam tangent (node1→node2).
        * ``'L'``           : (N_beams,)   — beam length.
        * ``'t_coil_start'``: (N_beams, 3) — coil tangent at node-1 attachment phi.
        * ``'t_coil_end'``  : (N_beams, 3) — coil tangent at node-2 for CC beams;
          zero vector for CF beams (foundation side has no coil tangent).
        """
        phis_start_cc = support_dofs['phis_start_cc']       # list[g] -> (n_beam_cc[g],)
        phis_end_cc   = support_dofs['phis_end_cc']         # list[g] -> (n_beam_cc[g],)
        phis_start_cf = support_dofs['phis_start_cf']       # list[i] -> (n_beam_cf[i],)
        x_foundation = support_dofs['x_foundation']       # list[i] -> (n_beam_cf[i], 3)

        # Each element appended here is a (n_beams_in_block, 3) array; lists
        # are assembled with jnp.concatenate at the end.  This issues one
        # gamma_eval per (curve, group) pair instead of one per beam.
        x_start_list, x_end_list = [], []
        t_coil_start_list, t_coil_end_list = [], []

        def append_cc_group(g):
            """Batch-append the CC beams of group ``g``."""
            n_g = self.n_beam_cc[g]
            if n_g == 0:
                return
            start_idx, end_idx, end_tfm = self._cc_groups[g]
            curve_s = curves[start_idx]
            curve_e = curves[end_idx]
            phi_s_g = phis_start_cc[g]                            # (n_g,)
            phi_e_g = phis_end_cc[g]                              # (n_g,)

            x_s_g   = curve_s.gamma_eval(phi_s_g)                # (n_g, 3)
            x_e_raw = curve_e.gamma_eval(phi_e_g)                # (n_g, 3)
            x_e_g   = self._apply_end_transform(x_e_raw, end_tfm)

            t_cs_raw = curve_s.gamma_eval(phi_s_g, diff_order=1) # (n_g, 3)
            t_cs_g   = t_cs_raw / (
                jnp.linalg.norm(t_cs_raw, axis=1, keepdims=True) + 1e-300
            )

            t_ce_raw = curve_e.gamma_eval(phi_e_g, diff_order=1) # (n_g, 3)
            t_ce_raw = self._apply_end_transform(t_ce_raw, end_tfm)
            t_ce_g   = t_ce_raw / (
                jnp.linalg.norm(t_ce_raw, axis=1, keepdims=True) + 1e-300
            )

            x_start_list.append(x_s_g)
            x_end_list.append(x_e_g)
            t_coil_start_list.append(t_cs_g)
            t_coil_end_list.append(t_ce_g)

        for i, curve_i in enumerate(curves):
            # ── CC beams of group i (start coil = i) ─────────────────────────
            append_cc_group(i)

            # ── CF beams for coil i ──────────────────────────────────────────
            n_cf_i = self.n_beam_cf[i]
            if n_cf_i > 0:
                phi_s_cf = phis_start_cf[i]                           # (n_cf_i,)
                x_s_cf   = curve_i.gamma_eval(phi_s_cf)              # (n_cf_i, 3)
                t_cs_raw = curve_i.gamma_eval(phi_s_cf, diff_order=1) # (n_cf_i, 3)
                t_cs_cf  = t_cs_raw / (
                    jnp.linalg.norm(t_cs_raw, axis=1, keepdims=True) + 1e-300
                )

                x_start_list.append(x_s_cf)
                x_end_list.append(x_foundation[i])                   # (n_cf_i, 3)
                t_coil_start_list.append(t_cs_cf)
                # CF: no coil tangent at foundation side → zero placeholder
                t_coil_end_list.append(jnp.zeros((n_cf_i, 3)))

        # ── Stellsym wrap group: coil 0 -> its phi = 0 image ─────────────────
        if self.stellsym:
            append_cc_group(self.n_base)

        x_start      = jnp.concatenate(x_start_list,      axis=0)  # (N, 3)
        x_end        = jnp.concatenate(x_end_list,        axis=0)  # (N, 3)
        t_coil_start = jnp.concatenate(t_coil_start_list, axis=0)  # (N, 3)
        t_coil_end   = jnp.concatenate(t_coil_end_list,   axis=0)  # (N, 3)

        diff = x_end - x_start                             # (N, 3)
        L    = jnp.linalg.norm(diff, axis=1)              # (N,)
        t_beam = diff / (L[:, None] + 1e-300)             # (N, 3)

        return {
            'x_start':      x_start,
            'x_end':        x_end,
            't_beam':       t_beam,
            'L':            L,
            't_coil_start': t_coil_start,
            't_coil_end':   t_coil_end,
        }

    def _direction_cosine_matrices(
        self,
        geom: dict,
        support_dofs: dict,
    ) -> jax.Array:
        """Compute per-beam 3×3 direction-cosine matrices.

        The beam local frame is defined as:

        * ``x_local = t_beam``  (unit vector node1→node2).
        * reference direction ``ref = cross(t_beam, t_coil_start)``
          (normal to the beam in the plane of beam + coil-start tangent).
        * ``z_local = normalize(Rodrigues(t_beam, thetas_orientation) @ ref)``
          — ``thetas_orientation`` rolls the cross-section about the beam axis.
        * ``y_local = cross(z_local, x_local)``

        Parameters
        ----------
        geom : dict
            Output of :meth:`_beam_geometry`.
        support_dofs : dict
            Must contain ``thetas_orientation_cc`` and ``thetas_orientation_cf``.

        Returns
        -------
        jax.Array, shape ``(N_beams, 3, 3)``
            Column-major: ``Gamma[b] = [x_local | y_local | z_local]``.
        """
        theta_cc = support_dofs['thetas_orientation_cc']  # list[g] -> (n_beam_cc[g],)
        theta_cf = support_dofs['thetas_orientation_cf']  # list[i] -> (n_beam_cf[i],)

        t_beam = geom['t_beam']           # (N, 3)
        t_coil = geom['t_coil_start']     # (N, 3)

        # Flatten theta angles into (N,) matching the beam ordering (per coil:
        # cc group i then cf coil i; stellsym wrap group appended last).
        # One concatenate per group/coil instead of one append per beam.
        theta_parts = []
        for i in range(self.n_base):
            if self.n_beam_cc[i] > 0:
                theta_parts.append(theta_cc[i])
            if self.n_beam_cf[i] > 0:
                theta_parts.append(theta_cf[i])
        if self.stellsym and self.n_beam_cc[self.n_base] > 0:
            theta_parts.append(theta_cc[self.n_base])
        thetas = jnp.concatenate(theta_parts, axis=0)  # (N,)

        def single_dcm(t_b, t_c, theta):
            # Reference direction: cross(beam-tangent, coil-tangent-at-start)
            ref = jnp.cross(t_b, t_c)
            ref_norm = jnp.linalg.norm(ref)
            # Guard against parallel tangents: fall back to a global-z reference
            ref = jnp.where(
                ref_norm > 1e-9,
                ref / ref_norm,
                jnp.array([0., 0., 1.]) - t_b * t_b[2],  # projection of z onto plane ⊥ t_b
            )
            ref = ref / (jnp.linalg.norm(ref) + 1e-300)

            # Roll the reference direction about the beam axis by theta
            R = _rodrigues(t_b, theta)
            z_local = R @ ref
            z_local = z_local / (jnp.linalg.norm(z_local) + 1e-300)

            x_local = t_b
            y_local = jnp.cross(z_local, x_local)
            y_local = y_local / (jnp.linalg.norm(y_local) + 1e-300)

            # Columns: x_local, y_local, z_local
            return jnp.stack([x_local, y_local, z_local], axis=1)  # (3, 3)

        return jax.vmap(single_dcm)(t_beam, t_coil, thetas)  # (N, 3, 3)

    def _endpoint_specs(
        self,
        geom: dict,
        gamma3: jax.Array,
    ) -> list[dict]:
        """Enumerate every coil-coupled beam endpoint.

        Returns one spec dict per coil-touching beam endpoint (CC beams yield
        two specs; CF beams yield one — the foundation side is handled
        separately in the spring assembly helpers).

        Each spec contains:

        * ``'b'``         : flat beam index.
        * ``'node_side'`` : ``0`` (node-1) or ``1`` (node-2).
        * ``'coil'``      : index of the base coil this endpoint couples to.
        * ``'coil_origin'``: group index into the per-group support-dofs
          lists (CC group index, or coil index for CF beams).
        * ``'j_local'``   : local beam index within that group's list entry.
        * ``'x_ep'``      : ``(3,)`` endpoint position in the frame where
          ``coil`` surface points live (after symmetry transform when required).
        * ``'gamma3'``    : ``(3, 3)`` DCM for this beam
          (columns: ``x_local``, ``y_local``, ``z_local``).
        * ``'sign_x'``    : ``True`` at node-1 (beam extends toward
          ``+x_local``); ``False`` at node-2.
        * ``'tfm'``       : symmetry transform tag ``'none'`` / ``'flip'`` /
          ``'flip_half'`` / ``'rotate'`` — applied to surface points before
          computing moment arms.  Always ``'none'`` for node-1.

        Parameters
        ----------
        geom : dict
            Output of :meth:`_beam_geometry`.
        gamma3 : jax.Array, shape ``(N_beams, 3, 3)``
            Output of :meth:`_direction_cosine_matrices`.

        Returns
        -------
        list of dicts, one per coil-touching endpoint.
        """
        specs = []

        def append_cc_group(g, b):
            """Append both endpoint specs of every CC beam in group ``g``."""
            start_idx, end_idx, end_tfm = self._cc_groups[g]
            for j in range(self.n_beam_cc[g]):
                g3 = gamma3[b]
                specs.append({
                    'b': b, 'coil_origin': g, 'j_local': j,
                    'node_side': 0, 'coil': start_idx,
                    'x_ep': geom['x_start'][b], 'gamma3': g3,
                    'sign_x': True, 'tfm': 'none',
                })
                specs.append({
                    'b': b, 'coil_origin': g, 'j_local': j,
                    'node_side': 1, 'coil': end_idx,
                    'x_ep': geom['x_end'][b], 'gamma3': g3,
                    'sign_x': False, 'tfm': end_tfm,
                })
                b += 1
            return b

        b = 0
        for i in range(self.n_base):
            b = append_cc_group(i, b)

            for j in range(self.n_beam_cf[i]):
                j_local = self.n_beam_cc[i] + j
                specs.append({
                    'b': b, 'coil_origin': i, 'j_local': j_local,
                    'node_side': 0, 'coil': i,
                    'x_ep': geom['x_start'][b], 'gamma3': gamma3[b],
                    'sign_x': True, 'tfm': 'none',
                })
                b += 1

        if self.stellsym:
            append_cc_group(self.n_base, b)

        return specs

    def _clamp_weights_for_spec(
        self,
        spec: dict,
        surf_pts: jax.Array,
        support_dofs,
    ):
        """Compute weights and moment arms for one beam endpoint.

        Applies the endpoint's symmetry transform to ``surf_pts``, projects
        the shifted surface points into the beam's local frame, and calls
        :attr:`attachment_fn`.

        Parameters
        ----------
        spec : dict
            One element from :meth:`_endpoint_specs`.
        surf_pts : jax.Array, shape ``(n_surf, 3)``
            Surface points of the *coupled* coil (``surface_pts_by_coil[spec['coil']]``).
        support_dofs : dict

        Returns
        -------
        w_k : jax.Array, shape ``(n_surf,)``
        r_k : jax.Array, shape ``(n_surf, 3)``
            Moment arms in the endpoint's local frame.
        """
        surf_tfm = self._apply_end_transform(surf_pts, spec['tfm'])
        r_k      = surf_tfm - spec['x_ep'][None, :]
        pts_beam = r_k @ spec['gamma3']
        # Slice this beam's cross-section scalar out of the ragged per-group
        # lists so attachment_fn receives the scalar for this specific beam.
        # The cross-section is indexed by the beam's originating *group*
        # ('coil_origin'; the node-2 CC endpoint couples to a different coil
        # than it originates from).
        if self._cross_section_dof_keys:
            i = spec['coil_origin']
            j = spec['j_local']
            beam_dofs = {
                **support_dofs,
                **{k: support_dofs[k][i][j] for k in self._cross_section_dof_keys},
            }
        else:
            beam_dofs = support_dofs
        w_k      = self.attachment_fn(pts_beam, beam_dofs, spec['sign_x'], self.beam_options)
        return w_k, r_k

    # ============================================================================
    # Stiffness helpers
    # ============================================================================

    def _local_stiffness(
        self,
        A: jax.Array,
        Iy: jax.Array,
        Iz: jax.Array,
        J: jax.Array,
        L: jax.Array,
    ) -> jax.Array:
        """Bisymmetric beam local stiffness (Eq. 4.34, McGuire et al.).

        Builds the 12×12 local stiffness matrix from four decoupled
        sub-problems: axial (EA/L), torsion (GJ/L), bending in x–y (EIz),
        and bending in x–z (EIy).  The sign convention follows Figure 4.6
        and Section §"Why the two bending planes carry opposite signs" of
        ``docs/theory/bisymbeam.rst``.

        Parameters
        ----------
        A, Iy, Iz, J : jax.Array, shape ``(N,)``
            Cross-section properties, one entry per beam.
        L : jax.Array, shape ``(N,)``
            Beam lengths.

        Returns
        -------
        jax.Array, shape ``(N, 12, 12)``
        """
        E, nu = self._beam_options['E'], self._beam_options['nu']
        G = E / (2.0 * (1.0 + nu))      # shear modulus

        # Convenience scalars per beam
        EAL  = E * A / L                  # axial stiffness factor
        GJL  = G * J / L                  # torsion stiffness factor
        EIz  = E * Iz                     # bending-xy pre-factor
        EIy  = E * Iy                     # bending-xz pre-factor
        L2   = L ** 2
        L3   = L ** 3

        def single_beam(EAL_b, GJL_b, EIz_b, EIy_b, L_b, L2_b, L3_b):
            # DOF ordering: [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2]
            # Indices:        0  1  2   3   4   5   6  7  8   9  10  11
            K = jnp.zeros((12, 12))

            # ── Axial (DOFs 0, 6) ───────────────────────────────────────────
            #  [  1  -1 ]
            #  [ -1   1 ] * EAL
            K = K.at[0, 0].set( EAL_b)
            K = K.at[0, 6].set(-EAL_b)
            K = K.at[6, 0].set(-EAL_b)
            K = K.at[6, 6].set( EAL_b)

            # ── Torsion (DOFs 3, 9) ─────────────────────────────────────────
            K = K.at[3, 3].set( GJL_b)
            K = K.at[3, 9].set(-GJL_b)
            K = K.at[9, 3].set(-GJL_b)
            K = K.at[9, 9].set( GJL_b)

            # ── Bending in x–y plane (DOFs 1,5,7,11 — v and θz) ───────────
            # Sub-block [v1, θz1, v2, θz2] with +6EIz/L² coupling terms
            # (positive sign: v = +θz * x in x–y plane)
            c12 = 12.0 * EIz_b / L3_b
            c6  =  6.0 * EIz_b / L2_b
            c4  =  4.0 * EIz_b / L_b
            c2  =  2.0 * EIz_b / L_b
            #  row/col 1 (v1)
            K = K.at[1, 1].set( c12);  K = K.at[1, 5].set( c6)
            K = K.at[1, 7].set(-c12);  K = K.at[1, 11].set( c6)
            #  row/col 5 (θz1)
            K = K.at[5, 1].set( c6);   K = K.at[5, 5].set( c4)
            K = K.at[5, 7].set(-c6);   K = K.at[5, 11].set( c2)
            #  row/col 7 (v2)
            K = K.at[7, 1].set(-c12);  K = K.at[7, 5].set(-c6)
            K = K.at[7, 7].set( c12);  K = K.at[7, 11].set(-c6)
            #  row/col 11 (θz2)
            K = K.at[11, 1].set( c6);  K = K.at[11, 5].set( c2)
            K = K.at[11, 7].set(-c6);  K = K.at[11, 11].set( c4)

            # ── Bending in x–z plane (DOFs 2,4,8,10 — w and θy) ───────────
            # Sub-block [w1, θy1, w2, θy2] with –6EIy/L² coupling terms
            # (negative sign: w = –θy * x in x–z plane, per sign convention)
            d12 = 12.0 * EIy_b / L3_b
            d6  =  6.0 * EIy_b / L2_b
            d4  =  4.0 * EIy_b / L_b
            d2  =  2.0 * EIy_b / L_b
            #  row/col 2 (w1)
            K = K.at[2, 2].set( d12);  K = K.at[2, 4].set(-d6)
            K = K.at[2, 8].set(-d12);  K = K.at[2, 10].set(-d6)
            #  row/col 4 (θy1)
            K = K.at[4, 2].set(-d6);   K = K.at[4, 4].set( d4)
            K = K.at[4, 8].set( d6);   K = K.at[4, 10].set( d2)
            #  row/col 8 (w2)
            K = K.at[8, 2].set(-d12);  K = K.at[8, 4].set( d6)
            K = K.at[8, 8].set( d12);  K = K.at[8, 10].set( d6)
            #  row/col 10 (θy2)
            K = K.at[10, 2].set(-d6);  K = K.at[10, 4].set( d2)
            K = K.at[10, 8].set( d6);  K = K.at[10, 10].set( d4)

            return K

        return jax.vmap(single_beam)(EAL, GJL, EIz, EIy, L, L2, L3)

    def _global_stiffness(
        self,
        K_local: jax.Array,
        Gamma_3: jax.Array,
    ) -> jax.Array:
        """Rotate local stiffness to global frame: ``Γ_12 K_local Γ_12^T``.

        ``Gamma_3`` holds the local axes as *columns*, so local coordinates
        of a global vector are ``v_loc = Γ^T v_glob`` and the congruence is
        ``K_glob = Γ K_local Γ^T``.

        Parameters
        ----------
        K_local : jax.Array, shape ``(N, 12, 12)``
        Gamma_3 : jax.Array, shape ``(N, 3, 3)``
            Per-beam 3×3 direction-cosine matrices.

        Returns
        -------
        jax.Array, shape ``(N, 12, 12)``
        """
        def rotate_one(K_loc, g3):
            G12 = _gamma_12(g3)       # (12, 12)
            return G12 @ K_loc @ G12.T

        return jax.vmap(rotate_one)(K_local, Gamma_3)

    # ============================================================================
    # Public geometry helper
    # ============================================================================

    def geometry(
        self,
        curves_jax: list,
        support_dofs: dict,
    ) -> dict:
        """Compute all per-beam geometry arrays in one traced call.

        Combines :meth:`_beam_geometry` (endpoint positions, lengths,
        tangents) and :meth:`_direction_cosine_matrices` (DCMs) so that
        callers can pass the result to :meth:`compute_weights`,
        :meth:`compute_attach`, :meth:`coo`, and :meth:`coupling_values` as
        the ``geom`` keyword argument, avoiding redundant re-computation.

        Parameters
        ----------
        curves_jax : list[CurveXYZFourierJAX]
            Traced base-coil centreline objects.
        support_dofs : dict
            Traced support DOF pytree.

        Returns
        -------
        dict
            All keys from :meth:`_beam_geometry` plus:

            * ``'gamma3'`` : jax.Array, shape ``(N_beams, 3, 3)`` — per-beam
              direction-cosine matrices (columns: local x/y/z axes).
        """
        geom = self._beam_geometry(curves_jax, support_dofs)
        geom['gamma3'] = self._direction_cosine_matrices(geom, support_dofs)
        return geom

    # ============================================================================
    # Support-ABC hook implementations
    # ============================================================================

    def compute_weights(
        self,
        coil_idx: int,
        surface_pts: jax.Array,
        curves_jax: list,
        dofs,
        geom: dict | None = None,
    ) -> jax.Array:
        """Winkler weights for coil ``coil_idx``.

        When ``dofs`` is ``None`` (e.g. during weight visualisation before a
        solve), delegates entirely to :class:`Support` (uniform ones when
        no ``fixed_clamp_fns`` were provided).  Otherwise sums the beam spring
        weights from all endpoints attached to ``coil_idx`` using the exact
        beam-frame geometry, then optionally adds the :class:`Support` term.

        Parameters
        ----------
        coil_idx : int
        surface_pts : jax.Array, shape ``(n_surf, 3)``
        curves_jax : list[CurveXYZFourierJAX]
            All base-coil centreline curves (traced).
        dofs : dict or None
        geom : dict or None
            Pre-computed geometry dict from :meth:`geometry`.  Computed
            internally when ``None``.

        Returns
        -------
        jax.Array, shape ``(n_surf,)``
        """
        if dofs is None:
            return Support.compute_weights(
                self, coil_idx, surface_pts, curves_jax, dofs
            )

        if geom is None:
            geom = self._beam_geometry(curves_jax, dofs)
        gamma3 = geom['gamma3'] if 'gamma3' in geom else self._direction_cosine_matrices(geom, dofs)
        specs  = self._endpoint_specs(geom, gamma3)

        w_total = jnp.zeros(surface_pts.shape[0])
        for spec in specs:
            if spec['coil'] == coil_idx:
                w_k, _ = self._clamp_weights_for_spec(spec, surface_pts, dofs)
                w_total = w_total + w_k

        if self._fixed_clamp_fns is not None:
            w_total = w_total + Support.compute_weights(
                self, coil_idx, surface_pts, curves_jax, dofs
            )
        return w_total

    def plot_support(
        self,
        fem,
        *,
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: dict | None = None,
        ax=None,
        s: float = 0.1,
        cmap: str = "viridis",
        color="C0",
        simple_mode: bool = False,
        beam_color="k",
        beam_lw: float = 1.5,
        **kwargs,
    ):
        """Scatter coil nodes by Winkler weight and overlay the support beams.

        Extends :meth:`Support.plot_support` by drawing every base-coil beam
        (coil-coil and coil-foundation) as a straight line segment between its
        two endpoints on the same axes.  No stellarator reflections or
        field-period rotations are added; only the base-coil beams returned by
        :meth:`_beam_geometry` are plotted.

        Parameters
        ----------
        fem : coil_fem.CoilFEM
            The FEM container owning this support (forwarded to
            :meth:`Support.plot_support`).
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``fem.base_curves_jax``.
        base_support_dofs : dict or None
            Per-coil support parameters.  Beams are drawn only when this is
            provided (the beam geometry needs the attachment phis and
            ``x_foundation``); ``None`` yields the scatter-only axes.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None
            Existing 3-D axes to draw on.  ``None`` (default) creates new axes.
        s : float
            Marker size for the scatter (default ``0.1``).
        cmap : str
            Colormap name for the support weights (default ``"viridis"``).
        color : color-like
            Marker colour used only when ``simple_mode`` is ``True``.
        simple_mode : bool
            Forwarded to :meth:`Support.plot_support`.
        beam_color : color-like
            Colour of the beam line segments (default ``"k"``).
        beam_lw : float
            Line width of the beam segments (default ``1.5``).
        **kwargs
            Extra keyword arguments forwarded to :meth:`ax.scatter`.

        Returns
        -------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            The 3-D axes used for the plot.
        """
        ax = super().plot_support(
            fem,
            base_curves_dofs=base_curves_dofs,
            base_support_dofs=base_support_dofs,
            ax=ax,
            s=s,
            cmap=cmap,
            color=color,
            simple_mode=simple_mode,
            **kwargs,
        )

        if base_support_dofs is None:
            return ax

        import numpy as onp
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in fem.base_curves_jax]

        curves_jax = [
            CurveXYZFourierJAX(c.quadpoints, base_curves_dofs[i], c.order)
            for i, c in enumerate(fem.base_curves_jax)
        ]
        geom = self._beam_geometry(curves_jax, base_support_dofs)
        x_s = onp.asarray(geom['x_start'], dtype=onp.float64)  # (N, 3)
        x_e = onp.asarray(geom['x_end'],   dtype=onp.float64)  # (N, 3)

        segments = onp.stack([x_s, x_e], axis=1)               # (N, 2, 3)
        ax.add_collection3d(
            Line3DCollection(segments, colors=beam_color, linewidths=beam_lw)
        )
        return ax

    def save_support_vtu(
        self,
        fem,
        out_dir: str = ".",
        *,
        prefix: str = "coil",
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: dict | None = None,
    ) -> list[str]:
        """Export support weights, full mesh, and the beam network as VTU files.

        Extends :meth:`Support.save_support_vtu` (per-coil weight meshes) by
        also writing, when ``base_support_dofs`` is provided:

        * ``{out_dir}/{prefix}_beams.vtu`` — VTU file containing every base-coil
          beam as a ``"line"`` cell.  Each line connects the two beam endpoints
          (``x_start`` / ``x_end`` from :meth:`_beam_geometry`).  Cell data
          fields:

          - ``beam_type`` — ``0`` for coil-coil (CC), ``1`` for coil-foundation (CF).
          - ``beam_length`` — rest-state beam length in metres.
          - ``coil_index`` — index of the originating base coil (0-based).

        No stellarator reflections or field-period rotations are added; only
        the base-coil beams are written.

        Parameters
        ----------
        fem : coil_fem.CoilFEM
            The FEM container owning this support.
        out_dir : str
            Output directory.  Created if it does not exist.
        prefix : str
            File-name prefix (default ``"coil"``).
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``fem.base_curves_jax``.
        base_support_dofs : dict or None
            Per-coil support parameters.  Beams are written only when this is
            provided (the beam geometry needs the attachment phis and
            ``x_foundation``).

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        written = super().save_support_vtu(
            fem, out_dir, prefix=prefix,
            base_curves_dofs=base_curves_dofs,
            base_support_dofs=base_support_dofs,
        )
        if base_support_dofs is None:
            return written

        import os
        import numpy as onp
        import meshio

        n_base = len(fem.base_curves_jax)
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in fem.base_curves_jax]

        curves_jax = [
            CurveXYZFourierJAX(c.quadpoints, base_curves_dofs[i], c.order)
            for i, c in enumerate(fem.base_curves_jax)
        ]
        geom = self._beam_geometry(curves_jax, base_support_dofs)

        x_s = onp.asarray(geom['x_start'], dtype=onp.float64)  # (N, 3)
        x_e = onp.asarray(geom['x_end'],   dtype=onp.float64)  # (N, 3)
        L   = onp.asarray(geom['L'],        dtype=onp.float64)  # (N,)
        n_beams = x_s.shape[0]

        # Build point array: first half = node-1 (start), second = node-2 (end).
        pts_beams = onp.concatenate([x_s, x_e], axis=0)        # (2N, 3)
        conn = onp.column_stack([                               # (N, 2)
            onp.arange(n_beams),
            onp.arange(n_beams, 2 * n_beams),
        ]).astype(onp.int32)

        # Cell labels: beam_type (CC=0, CF=1) and originating coil index.
        # Beams are ordered coil-major, cc-then-cf, with per-coil counts;
        # the stellsym wrap group (originating at coil 0) is appended last.
        coil_idx_arr = onp.concatenate([
            onp.full(self.n_beam_cc[i] + self.n_beam_cf[i], i, dtype=onp.int32)
            for i in range(n_base)
        ]) if n_base else onp.zeros(0, dtype=onp.int32)
        beam_type = onp.concatenate([
            onp.concatenate([
                onp.zeros(self.n_beam_cc[i], dtype=onp.int32),
                onp.ones(self.n_beam_cf[i], dtype=onp.int32),
            ])
            for i in range(n_base)
        ]) if n_base else onp.zeros(0, dtype=onp.int32)
        if self.stellsym:
            n_wrap = self.n_beam_cc[n_base]
            coil_idx_arr = onp.concatenate(
                [coil_idx_arr, onp.zeros(n_wrap, dtype=onp.int32)])
            beam_type = onp.concatenate(
                [beam_type, onp.zeros(n_wrap, dtype=onp.int32)])

        beam_path = os.path.join(out_dir, f"{prefix}_beams.vtu")
        meshio.Mesh(
            points=pts_beams,
            cells=[("line", conn)],
            cell_data={
                "beam_type":   [beam_type],
                "beam_length": [L],
                "coil_index":  [coil_idx_arr],
            },
        ).write(beam_path)
        written.append(beam_path)

        return written

    def compute_attach(
        self,
        coil_idx: int,
        surface_pts: jax.Array,
        curves_jax: list,
        dofs,
        state: dict,
        geom: dict | None = None,
    ) -> jax.Array:
        """Weighted-average beam endpoint displacement at coil surface nodes.

        For each beam endpoint spec attached to coil ``coil_idx``, computes
        the beam displacement at each surface node ``k`` as

        .. math::

            u_{\\text{beam},k}^b = u_{\\text{endpoint}}^b
                + \\theta_{\\text{endpoint}}^b \\times r_k^b

        where :math:`r_k^b` is the moment arm in the endpoint's frame
        (surface points shifted by the symmetry transform and projected
        relative to the endpoint).  The result is mapped back to the coil
        frame using the inverse symmetry transform.

        Parameters
        ----------
        coil_idx : int
        surface_pts : jax.Array, shape ``(n_surf, 3)``
        curves_jax : list[CurveXYZFourierJAX]
        dofs : dict
        state : dict
            Must contain ``'u_s'`` : jax.Array, shape ``(n_support_dofs,)``.
        geom : dict or None
            Pre-computed geometry dict from :meth:`geometry`.  Computed
            internally when ``None``.

        Returns
        -------
        jax.Array, shape ``(n_surf, 3)``
        """
        u_s    = state['u_s']                            # (12 * N_beams,)
        u_beams = u_s.reshape(self.n_beams_total, 12)   # (N_beams, 12)
        n_surf  = surface_pts.shape[0]

        if geom is None:
            geom = self._beam_geometry(curves_jax, dofs)
        gamma3 = geom['gamma3'] if 'gamma3' in geom else self._direction_cosine_matrices(geom, dofs)
        specs  = self._endpoint_specs(geom, gamma3)

        w_total      = jnp.zeros(n_surf)
        u_attach_num = jnp.zeros((n_surf, 3))

        for spec in specs:
            if spec['coil'] != coil_idx:
                continue

            b          = spec['b']
            node_side  = spec['node_side']
            t_off      = 6 * node_side

            w_k, r_k   = self._clamp_weights_for_spec(spec, surface_pts, dofs)
            u_ep       = u_beams[b, t_off     : t_off + 3]
            theta_ep   = u_beams[b, t_off + 3 : t_off + 6]

            # Rigid-body displacement in the endpoint frame
            u_beam_k = u_ep[None, :] + jax.vmap(
                lambda r: jnp.cross(theta_ep, r))(r_k)

            # Map back to coil frame
            u_beam_k = self._apply_end_transform(u_beam_k, spec['tfm'], inverse=True)

            w_total      = w_total + w_k
            u_attach_num = u_attach_num + w_k[:, None] * u_beam_k

        w_safe   = jnp.where(w_total > 1e-300, w_total, jnp.ones_like(w_total))
        u_attach = jnp.where(
            (w_total > 1e-300)[:, None],
            u_attach_num / w_safe[:, None],
            jnp.zeros((n_surf, 3), dtype=surface_pts.dtype),
        )
        return u_attach

    def coupling_pattern(
        self,
        coil_dof_offsets: list[int],
        support_dof_offset: int,
        surface_node_indices_by_coil: list,
    ) -> tuple:
        """Static numpy I/J index arrays for the off-diagonal K_cs / K_sc blocks.

        All arithmetic operates on static (host) integer data; no JAX tracing
        occurs.  The returned arrays are in the same entry order as
        :meth:`coupling_values`, so the two can be combined into a COO triplet
        by pairing them element-wise.

        Parameters
        ----------
        coil_dof_offsets : list[int]
            Global DOF offset for each coil in the merged system.
        support_dof_offset : int
            Global DOF offset for the support DOFs.
        surface_node_indices_by_coil : list[np.ndarray]
            Per-coil ``(n_surf_i,)`` integer arrays mapping compact surface
            index → global mesh node index.

        Returns
        -------
        I_cs, J_cs, I_sc, J_sc : np.ndarray, shape ``(nnz_cs,)`` / ``(nnz_sc,)``
        """
        d3 = np.arange(3, dtype=np.int32)

        I_cs_parts, J_cs_parts = [], []
        I_sc_parts, J_sc_parts = [], []

        def _add_endpoint(b: int, node_side: int, coil_i: int) -> None:
            surf_idx = np.asarray(surface_node_indices_by_coil[coil_i], dtype=np.int32)
            n_surf   = surf_idx.shape[0]
            t_off    = 6 * node_side
            r_off    = 6 * node_side + 3

            beam_trans_base = support_dof_offset + 12 * b + t_off
            beam_rot_base   = support_dof_offset + 12 * b + r_off

            coil_dof_base   = coil_dof_offsets[coil_i] + 3 * surf_idx  # (n_surf,)
            coil_dofs       = coil_dof_base[:, None] + d3[None, :]      # (n_surf, 3)
            beam_trans_dofs = (beam_trans_base + d3).astype(np.int32)   # (3,)
            beam_rot_dofs   = (beam_rot_base   + d3).astype(np.int32)   # (3,)

            rows_cs   = np.broadcast_to(coil_dofs[:, :, None],          (n_surf, 3, 3))
            cols_cs_t = np.broadcast_to(beam_trans_dofs[None, None, :], (n_surf, 3, 3))
            cols_cs_r = np.broadcast_to(beam_rot_dofs[None, None, :],   (n_surf, 3, 3))
            I_cs_parts.append(rows_cs.reshape(-1).copy())
            I_cs_parts.append(rows_cs.reshape(-1).copy())
            J_cs_parts.append(cols_cs_t.reshape(-1).copy())
            J_cs_parts.append(cols_cs_r.reshape(-1).copy())

            rows_sc_t = np.broadcast_to(beam_trans_dofs[None, :, None], (n_surf, 3, 3))
            rows_sc_r = np.broadcast_to(beam_rot_dofs[None, :, None],   (n_surf, 3, 3))
            cols_sc   = np.broadcast_to(coil_dofs[:, None, :],          (n_surf, 3, 3))
            I_sc_parts.append(rows_sc_t.reshape(-1).copy())
            I_sc_parts.append(rows_sc_r.reshape(-1).copy())
            J_sc_parts.append(cols_sc.reshape(-1).copy())
            J_sc_parts.append(cols_sc.reshape(-1).copy())

        b = 0
        for i in range(self.n_base):
            start_i, end_i, _ = self._cc_groups[i]
            for _j in range(self.n_beam_cc[i]):
                _add_endpoint(b, 0, start_i)
                _add_endpoint(b, 1, end_i)
                b += 1
            for _j in range(self.n_beam_cf[i]):
                _add_endpoint(b, 0, i)
                b += 1

        if self.stellsym:
            start_w, end_w, _ = self._cc_groups[self.n_base]
            for _j in range(self.n_beam_cc[self.n_base]):
                _add_endpoint(b, 0, start_w)
                _add_endpoint(b, 1, end_w)
                b += 1

        if not I_cs_parts:
            empty = np.zeros(0, dtype=np.int32)
            return empty, empty, empty, empty

        return (
            np.concatenate(I_cs_parts).astype(np.int32),
            np.concatenate(J_cs_parts).astype(np.int32),
            np.concatenate(I_sc_parts).astype(np.int32),
            np.concatenate(J_sc_parts).astype(np.int32),
        )

    def coupling_values(
        self,
        curves_jax: list,
        support_dofs: dict,
        surface_pts_by_coil: list,
        geom: dict | None = None,
    ) -> tuple:
        """Traced V arrays for the off-diagonal K_cs / K_sc blocks.

        Computes one pair of value blocks ``(blk_t, blk_r)`` per
        coil-touching beam endpoint, in the same order as
        :meth:`coupling_pattern`.  All outputs are traced through
        ``curves_jax`` and ``support_dofs``.

        Parameters
        ----------
        curves_jax : list[CurveXYZFourierJAX]
            Traced base-coil centreline objects.
        support_dofs : dict
            Traced support DOF pytree.
        surface_pts_by_coil : list[jax.Array]
            Per-coil surface node positions.
        geom : dict or None
            Pre-computed beam geometry dict (output of :meth:`_beam_geometry`
            and optionally including ``'gamma3'``).  If ``None``, geometry is
            computed internally.

        Returns
        -------
        V_cs : jax.Array, shape ``(nnz_cs,)``
        V_sc : jax.Array, shape ``(nnz_sc,)``
        """
        if geom is None:
            geom = self._beam_geometry(curves_jax, support_dofs)
        gamma3 = geom['gamma3'] if 'gamma3' in geom else self._direction_cosine_matrices(geom, support_dofs)
        specs = self._endpoint_specs(geom, gamma3)

        V_cs_parts: list = []
        V_sc_parts: list = []

        for spec in specs:
            coil_i   = spec['coil']
            surf_pts = surface_pts_by_coil[coil_i]
            w_k, r_k = self._clamp_weights_for_spec(spec, surf_pts, support_dofs)

            Q    = self._tfm_Q[spec['tfm']]
            Qinv = Q.T

            skew_r = jax.vmap(_skew)(r_k)              # (n_surf, 3, 3)
            wk     = w_k[:, None, None]                 # (n_surf, 1, 1)

            blk_t_cs = (-self.k_lin) * wk * Qinv[None]
            blk_r_cs = ( self.k_lin) * wk * (Qinv[None] @ skew_r)
            V_cs_parts.append(blk_t_cs.reshape(-1))
            V_cs_parts.append(blk_r_cs.reshape(-1))

            blk_t_sc = (-self.k_lin) * wk * Q[None]
            blk_r_sc = (-self.k_tor) * wk * (skew_r @ Q[None])
            V_sc_parts.append(blk_t_sc.reshape(-1))
            V_sc_parts.append(blk_r_sc.reshape(-1))

        if not V_cs_parts:
            return jnp.zeros(0), jnp.zeros(0)

        return jnp.concatenate(V_cs_parts), jnp.concatenate(V_sc_parts)

    def coupling_terms(
        self,
        curves_jax: list,
        support_dofs: dict,
        surface_pts_by_coil: list,
        coil_dof_offsets: list,
        support_dof_offset: int,
        surface_node_indices_by_coil: list,
    ) -> dict:
        """COO triplets for the off-diagonal coupling blocks ``K_cs`` / ``K_sc``.

        **Physics convention:**
        For a spring connecting beam endpoint ``b`` (translation DOF ``u_b``,
        rotation DOF ``θ_b``) to coil surface node ``k`` with weight ``w_k^b``
        and moment-arm ``r_k^b`` (in the endpoint's local frame):

        .. math::

            F_{\\text{coil},k} = -k_{\\text{lin}} w_k^b (u_b + θ_b × r_k^b - u_k)

        ``K_cs`` (row = coil translation DOF of node ``k``,
        col = beam endpoint DOF of beam ``b``):

        * Translation block: ``-k_{\\text{lin}} w_k^b Q^{-1}``
        * Rotation block:    ``+k_{\\text{lin}} w_k^b Q^{-1} [r_k^b]×``
          (arising from ``θ × r = -[r]× θ``, sign folded)

        ``K_sc`` (row = beam endpoint DOF, col = coil translation DOF):

        * Beam translation, coil translation: ``-k_{\\text{lin}} w_k^b Q``
        * Beam rotation, coil translation:    ``-k_{\\text{tor}} w_k^b [r_k^b]× Q``
          (mesh column of ``τ = k_tor Σ w r × (u_att − u_mesh)``, the
          transpose-free counterpart of the ``+k_tor Σ w [r]×`` endpoint
          columns in ``K_ss``)

        where ``Q`` is the endpoint's symmetry transform (identity for
        untransformed endpoints).  For a ghost endpoint the beam couples to
        the *image* mesh (``u_image = Q u_coil``), and the coil rows carry
        the mirrored partner-beam force (the ``Q^{-1}``-image of the ghost
        interaction) — see the class docstring.

        Parameters
        ----------
        curves_jax : list[CurveXYZFourierJAX]
            Traced base-coil centreline objects.
        support_dofs : dict
        surface_pts_by_coil : list[jax.Array]
            ``(n_surf_i, 3)`` per coil.
        coil_dof_offsets : list[int]
            Global DOF offset for coil ``i`` in the merged system.
        support_dof_offset : int
            Global DOF offset for support DOFs in the merged system.
        surface_node_indices_by_coil : list[np.ndarray]
            ``(n_surf_i,)`` integer arrays mapping compact surface index → global
            mesh node index, used to build row/column DOF indices.

        Returns
        -------
        dict with keys ``'I_cs'``, ``'J_cs'``, ``'V_cs'``,
        ``'I_sc'``, ``'J_sc'``, ``'V_sc'``.
        """
        I_cs, J_cs, I_sc, J_sc = self.coupling_pattern(
            coil_dof_offsets, support_dof_offset, surface_node_indices_by_coil,
        )
        V_cs, V_sc = self.coupling_values(
            curves_jax, support_dofs, surface_pts_by_coil,
        )
        return {
            'I_cs': jnp.asarray(I_cs), 'J_cs': jnp.asarray(J_cs), 'V_cs': V_cs,
            'I_sc': jnp.asarray(I_sc), 'J_sc': jnp.asarray(J_sc), 'V_sc': V_sc,
        }

    # ============================================================================
    # Spring coupling helpers
    # ============================================================================

    def _endpoint_weights_and_r(
        self,
        curves: list[CurveXYZFourierJAX],
        geom: dict,
        gamma3: jax.Array,
        support_dofs: dict,
        surface_pts_by_coil: list[jax.Array],
    ) -> list[dict]:
        """Per-beam endpoint spring weights and moment arms.

        Returns a list of length ``N_beams``.  Each element is a list of
        endpoint dicts (two per CC beam, two per CF beam) with keys:

        * ``'w'``           — weight array ``(n_surf,)`` or scalar ``1.0``.
        * ``'r'``           — moment-arm array ``(n_surf, 3)`` or ``(3,)``.
        * ``'node_side'``   — ``0`` or ``1``.
        * ``'coil'``        — base-coil index (absent for foundation entries).
        * ``'tfm'``         — symmetry transform tag (absent for foundation
          entries).
        * ``'is_foundation'``— present and ``True`` for CF node-2 entries.

        Parameters
        ----------
        curves : list[CurveXYZFourierJAX]
        geom : dict  — output of :meth:`_beam_geometry`.
        gamma3 : jax.Array, shape ``(N_beams, 3, 3)``
            Output of :meth:`_direction_cosine_matrices`, passed in from
            :meth:`coo` so it is computed only once.
        support_dofs : dict
        surface_pts_by_coil : list[jax.Array]
            One ``(n_surf_i, 3)`` array per base coil.

        Returns
        -------
        list of length ``N_beams``, each element a list of endpoint dicts.
        """
        x_foundation  = support_dofs['x_foundation']
        specs         = self._endpoint_specs(geom, gamma3)

        # Group coil-side specs by beam index
        beam_eps: list[list] = [[] for _ in range(self.n_beams_total)]
        for spec in specs:
            surf_pts = surface_pts_by_coil[spec['coil']]
            w_k, r_k = self._clamp_weights_for_spec(spec, surf_pts, support_dofs)
            beam_eps[spec['b']].append({
                'w': w_k, 'r': r_k,
                'node_side': spec['node_side'],
                'coil': spec['coil'],
                'tfm': spec['tfm'],
            })

        # Append foundation entries for CF beams (node-2 side, no clamp)
        for i in range(self.n_base):
            for j in range(self.n_beam_cf[i]):
                b      = self._beam_offsets[i] + self.n_beam_cc[i] + j
                x_ep   = geom['x_end'][b]                  # = x_foundation[i][j]
                r_fnd  = x_foundation[i][j] - x_ep         # ~zero at rest
                beam_eps[b].append({
                    'w': 1.0, 'r': r_fnd,
                    'node_side': 1, 'is_foundation': True,
                })

        return beam_eps

    def _spring_stiffness_contributions(
        self,
        beam_endpoints: list[list[dict]],
    ) -> jax.Array:
        """Per-beam 12×12 spring stiffness blocks (endpoint-diagonal contribution).

        Computes the support-local half of the spring coupling stiffness,
        i.e. the part that couples beam endpoint DOFs to themselves.  The
        coil-side and cross-DOF terms belong to the future ``coupling_terms()``
        method.

        The spring couples the *attach* displacement
        ``u_att,i = u_ep + θ × r_i`` to the mesh: force
        ``F = k_lin Σ_i w_i (u_att,i − u_mesh,i)`` and torque
        ``τ = k_tor Σ_i w_i r_i × (u_att,i − u_mesh,i)``.  For each endpoint
        at node ``n`` (local DOFs ``6n:6n+3`` translation, ``6n+3:6n+6``
        rotation) this yields four blocks:

        * **Translation–translation**::

              K[t, t] += (Σ_i w_i) * k_lin * I_3

        * **Translation–rotation** (from ``θ × r = -[r]× θ``)::

              K[t, r] += -k_lin * Σ_i (w_i * [r_i]×)

        * **Torque–translation**::

              K[r, t] += k_tor * Σ_i (w_i * [r_i]×)

        * **Torque–rotation** (from ``r × (θ × r)``; positive semidefinite —
          this is what pins the beam's axial-torsion rigid-body mode)::

              K[r, r] += -k_tor * Σ_i (w_i * [r_i]× [r_i]×)

        NOTE: unless ``k_tor == k_lin`` the torque rows are not the
        transpose of the translation rows, so K_ss is not symmetric.  A
        future cuDSS monolithic path must use mtype_id=0.

        For CF foundation side, ``w`` is scalar 1.0 and ``r`` is a single
        ``(3,)`` vector.

        Parameters
        ----------
        beam_endpoints : list
            Output of :meth:`_endpoint_weights_and_r`.

        Returns
        -------
        jax.Array, shape ``(N_beams, 12, 12)``
        """
        K_spring_list = []

        for ep_list in beam_endpoints:
            K = jnp.zeros((12, 12))

            for ep in ep_list:
                w   = ep['w']
                r   = ep['r']
                n   = ep['node_side']          # 0 or 1
                t_off = 6 * n                  # translation DOF offset (0 or 6)
                r_off = 6 * n + 3              # rotation DOF offset (3 or 9)
                is_found = ep.get('is_foundation', False)

                if is_found:
                    # Foundation side: scalar w=1, single moment arm r (3,)
                    w_sum = 1.0
                    skew_sum = _skew(r)        # (3, 3)
                    skew2_sum = skew_sum @ skew_sum
                else:
                    # Coil side: w is (n_surf,), r is (n_surf, 3)
                    w_sum = jnp.sum(w)         # scalar
                    skews = jax.vmap(_skew)(r)                       # (n, 3, 3)
                    # Weighted sums: Σ w_i [r_i]× and Σ w_i [r_i]×[r_i]×
                    skew_sum  = jnp.einsum('n,nij->ij', w, skews)    # (3, 3)
                    skew2_sum = jnp.einsum('n,nij->ij', w,
                                           skews @ skews)            # (3, 3)

                # Translation–translation: (Σ w_i) * k_lin * I_3
                K_tt = (self.k_lin * w_sum) * jnp.eye(3)
                K = K.at[t_off:t_off+3, t_off:t_off+3].add(K_tt)

                # Translation–rotation: -k_lin * Σ w_i [r_i]×  (θ × r term)
                K_tr = -self.k_lin * skew_sum  # (3, 3)
                K = K.at[t_off:t_off+3, r_off:r_off+3].add(K_tr)

                # Torque–translation: k_tor * Σ w_i [r_i]×
                K_rt = self.k_tor * skew_sum   # (3, 3)
                K = K.at[r_off:r_off+3, t_off:t_off+3].add(K_rt)

                # Torque–rotation: -k_tor * Σ w_i [r_i]×[r_i]×  (PSD; pins
                # the axial-torsion rigid-body mode of the isolated beam)
                K_rr = -self.k_tor * skew2_sum  # (3, 3)
                K = K.at[r_off:r_off+3, r_off:r_off+3].add(K_rr)

            K_spring_list.append(K)

        return jnp.stack(K_spring_list, axis=0)  # (N, 12, 12)

    # ============================================================================
    # Scatter helper
    # ============================================================================

    def _scatter_block_diagonal(self, K_beam: jax.Array) -> tuple:
        """Flatten ``(N, 12, 12)`` block-diagonal stiffness into COO triplets.

        Row/column indices ``I``, ``J`` are pre-built at construction (static);
        only the value array ``V`` is traced.

        Parameters
        ----------
        K_beam : jax.Array, shape ``(N_beams, 12, 12)``

        Returns
        -------
        I : np.ndarray, shape ``(N*144,)``   — static int32 row indices
        J : np.ndarray, shape ``(N*144,)``   — static int32 column indices
        V : jax.Array, shape ``(N*144,)``    — traced float values
        """
        V = K_beam.reshape(-1)
        return self._coo_I, self._coo_J, V

    # ============================================================================
    # coo — the primary assembly entry point
    # ============================================================================

    def coo(
        self,
        curves_jax: list,
        support_dofs: dict,
        surface_pts_by_coil: list[jax.Array] | None = None,
        geom: dict | None = None,
    ) -> tuple:
        """Return the support-local stiffness block ``K_ss`` in COO format.

        All outputs except ``I`` and ``J`` are differentiable w.r.t.
        ``curves_jax`` and ``support_dofs``.

        Parameters
        ----------
        curves_jax : list[CurveXYZFourierJAX]
            Traced base-coil centreline objects, one per base coil.
        support_dofs : dict
            Traced support DOF pytree containing ``phis_start_cc``,
            ``phis_end_cc``, ``phis_start_cf``, ``x_foundation``,
            ``thetas_orientation_cc``, ``thetas_orientation_cf``, and any
            keys required by ``cross_section_fn`` and ``attachment_fn``.
        surface_pts_by_coil : list[jax.Array] or None
            Current surface node positions per coil, shape
            ``(n_surface_nodes_i, 3)`` for coil ``i``.  Required when
            ``n_beam_cc > 0`` or ``n_beam_cf > 0`` (i.e. always for real
            use).  If ``None``, spring contributions are skipped and each
            beam's 12×12 block contains only the bare beam stiffness — the
            matrix will have rank 6 per block and is **singular**.
        geom : dict or None
            Pre-computed geometry dict from :meth:`geometry`.  Computed
            internally when ``None``.

        Returns
        -------
        I : np.ndarray, shape ``(N*144,)``  — static row indices
        J : np.ndarray, shape ``(N*144,)``  — static column indices
        V : jax.Array,  shape ``(N*144,)``  — stiffness values (traced)
        n_dofs : int                         — total support DOFs
        """
        if geom is None:
            geom = self._beam_geometry(curves_jax, support_dofs)
        Gamma_3 = geom['gamma3'] if 'gamma3' in geom else self._direction_cosine_matrices(geom, support_dofs)

        A_all, Iy_all, Iz_all, J_all = self.cross_section_fn(support_dofs)
        A  = jnp.concatenate([jnp.atleast_1d(a) for a in A_all])
        Iy = jnp.concatenate([jnp.atleast_1d(a) for a in Iy_all])
        Iz = jnp.concatenate([jnp.atleast_1d(a) for a in Iz_all])
        Jj = jnp.concatenate([jnp.atleast_1d(a) for a in J_all])

        K_local  = self._local_stiffness(A, Iy, Iz, Jj, geom['L'])
        K_global = self._global_stiffness(K_local, Gamma_3)

        if surface_pts_by_coil is not None:
            beam_endpoints = self._endpoint_weights_and_r(
                curves_jax, geom, Gamma_3, support_dofs, surface_pts_by_coil
            )
            K_spring = self._spring_stiffness_contributions(beam_endpoints)
            K_beam = K_global + K_spring
        else:
            K_beam = K_global

        I, J, V = self._scatter_block_diagonal(K_beam)
        return I, J, V, self.n_support_dofs

    # ============================================================================
    # solve — standalone forward solve
    # ============================================================================

    def _assemble_rhs(
        self,
        geom: dict,
        beam_endpoints: list[list[dict]],
        u_mesh_by_coil: list[jax.Array] | None,
    ) -> jax.Array:
        """Assemble the beam-network right-hand side vector.

        Parameters
        ----------
        geom : dict
            Output of :meth:`_beam_geometry`.
        beam_endpoints : list
            Output of :meth:`_endpoint_weights_and_r`.
        u_mesh_by_coil : list[jax.Array] or None
            Per-coil surface mesh displacements ``(n_surface_nodes_i, 3)``.
            ``None`` means all mesh displacements are zero (e.g. initial state).

        Returns
        -------
        jax.Array, shape ``(n_support_dofs,)``
        """
        f = jnp.zeros(self.n_support_dofs)

        for b, ep_list in enumerate(beam_endpoints):
            for ep in ep_list:
                n     = ep['node_side']
                t_off = 6 * n       # translation DOF start
                r_off = 6 * n + 3  # rotation DOF start
                is_found = ep.get('is_foundation', False)
                dof_base = 12 * b

                if is_found:
                    # Foundation side: RHS from zero-displacement-at-rest
                    # (x_foundation is already the endpoint position at rest,
                    # so r_foundation ≈ 0 and the RHS contribution is zero
                    # unless x_foundation has drifted — handled by K_ss * u_s = f).
                    pass
                else:
                    # Coil side: mesh-displacement force/torque targets,
                    # equal to -K_sc @ u_mesh blockwise (see coupling_terms).
                    # Ghost endpoints couple to the *image* mesh, so the
                    # per-node displacement is Q u_mesh (Q = I when 'none').
                    if u_mesh_by_coil is not None:
                        w      = ep['w']                           # (n_surf,)
                        r      = ep['r']                           # (n_surf, 3)
                        Q      = self._tfm_Q[ep['tfm']]            # (3, 3)
                        u_mesh = u_mesh_by_coil[ep['coil']]        # (n_surf, 3)
                        um     = u_mesh @ Q.T                      # Q u_mesh per node
                        # f_t += k_lin * Σ w_i (Q u_mesh_i)
                        f_trans = self.k_lin * jnp.einsum('n,nd->d', w, um)
                        f = f.at[dof_base + t_off: dof_base + t_off + 3].add(f_trans)
                        # f_r += k_tor * Σ w_i (r_i × (Q u_mesh_i)) — matches
                        # the -k_tor w [r]× Q columns of K_sc.
                        f_rot = self.k_tor * jnp.einsum(
                            'n,nd->d', w, jnp.cross(r, um))
                        f = f.at[dof_base + r_off: dof_base + r_off + 3].add(f_rot)

        return f

    def solve(self, inputs: dict) -> dict:
        """Standalone forward solve for beam DOFs.

        Uses ``lineax`` for a differentiable dense linear solve (implicit VJP
        through the factorization is built into ``lineax`` — no custom VJP
        needed here).

        Parameters
        ----------
        inputs : dict
            Must contain:

            * ``'curves_jax'`` : list[CurveXYZFourierJAX]
            * ``'support_dofs'``     : dict
            * ``'surface_pts_by_coil'`` : list[jax.Array] — current surface
              node positions per coil, each ``(n_surface_nodes_i, 3)``.

            Optional:

            * ``'u_mesh_by_coil'`` : list[jax.Array] or None — coil-side
              mesh displacements ``(n_surface_nodes_i, 3)``.  Defaults to
              zero (uncoupled, i.e. beams find equilibrium against a fixed
              coil surface).

        Returns
        -------
        dict with keys:

        * ``'u_s'``             : ``(n_support_dofs,)`` — beam DOF solution.
        * ``'curves_jax'``      : echoed back for driver convenience.
            * ``'support_dofs'``    : echoed back for driver convenience.
        """
        curves_jax          = inputs['curves_jax']
        support_dofs        = inputs['support_dofs']
        surface_pts_by_coil = inputs['surface_pts_by_coil']
        u_mesh_by_coil      = inputs.get('u_mesh_by_coil', None)

        # Use pre-computed geometry from driver when available, else compute.
        geom = inputs.get('geom') or self.geometry(curves_jax, support_dofs)
        gamma3 = geom['gamma3']

        # Assemble K_ss (traced) — pass pre-computed geometry.
        I, J, V, n_dofs = self.coo(
            curves_jax, support_dofs, surface_pts_by_coil, geom=geom
        )

        # Build dense K_ss from COO.  n_support_dofs is small in practice
        # (e.g. 5 coils × 6 beams × 12 DOFs = 360), so dense is fine.
        K_ss = jnp.zeros((n_dofs, n_dofs))
        K_ss = K_ss.at[I, J].add(V)

        # Assemble RHS — reuse geometry.
        beam_endpoints = self._endpoint_weights_and_r(
            curves_jax, geom, gamma3, support_dofs, surface_pts_by_coil
        )
        f_s = self._assemble_rhs(geom, beam_endpoints, u_mesh_by_coil)

        # Solve K_ss u_s = f_s using lineax (handles non-symmetric K).
        operator = lineax.MatrixLinearOperator(K_ss)
        solution = lineax.linear_solve(operator, f_s, solver=lineax.LU())
        u_s = solution.value

        return {
            'u_s':          u_s,
            'curves_jax':   curves_jax,
            'support_dofs': support_dofs,
        }
