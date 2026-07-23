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
from ..geo.symmetries import _flip_points, _rotate_points_z


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
    base_curves_jax : list[CurveXYZFourierJAX]
        Base coil centreline curves (before symmetry expansion).  Used only
        for static metadata (``quadpoints``, ``order``) at construction;
        traced curves are always rebuilt inside :meth:`coo` / :meth:`solve`.
    beam_options : dict
        Contains the following entries
        n_beam_cc : int
            Number of coil-coil beams per base coil (connecting coil ``i`` to
            coil ``i+1``, with symmetry-aware wraparound at the last coil).
        n_beam_cf : int
            Number of coil-foundation beams per base coil.
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
        returned array has shape ``(n_base, n_beam_cc + n_beam_cf)``.
        Section-property validation is delegated to this callable.
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

    * Beams are indexed ``b = i * (n_beam_cc + n_beam_cf) + j``, with CC
      beams ``j = 0 … n_beam_cc-1`` and CF beams
      ``j = n_beam_cc … n_beam_cc+n_beam_cf-1`` within coil ``i``.
    * Local DOF ordering per beam follows Figure 4.6 of
      ``docs/theory/bisymbeam.rst``:
      ``[u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]``.

    **Wraparound for the last coil's CC beams**:

    * ``i < n_base - 1``: endpoint attaches to ``base_coil[i+1]`` without
      any transform.
    * ``i == n_base - 1``, ``stellsym=True``: attaches to a stellarator-
      symmetric image of ``base_coil[n_base-1]`` (flip ``y, z`` signs).
    * ``i == n_base - 1``, ``stellsym=False``: attaches to the first
      field-period rotation of ``base_coil[0]`` (rotate by
      ``2π / nfp`` about z).

    **Stiffness matrix symmetry**: the torque law
    ``τ = k_tor (u_endpoint − u_mesh) × r`` contributes an asymmetric
    ``[r]×`` block.  ``K_ss`` is therefore **not symmetric** in general.
    :meth:`solve` uses ``lineax`` which imposes no symmetry assumption.
    A future monolithic cuDSS path must use ``mtype_id=0`` (general).
    """

    def __init__(
        self,
        nfp:int,
        stellsym:bool,
        beam_options: dict,
        base_curves_jax: list[CurveXYZFourierJAX],
        cross_section_fn: Callable,
        attachment_fn: Callable,
        cross_section_dof_keys: tuple = (),
        fixed_clamp_fns=None,
    ):
        super().__init__(fixed_clamp_fns=fixed_clamp_fns)

        self.base_curves_jax = list(base_curves_jax)
        self._beam_options = beam_options
        self._nfp = nfp
        self._stellsym = stellsym
        self._n_beam_cc = int(beam_options['n_beam_cc'])
        self._n_beam_cf = int(beam_options['n_beam_cf'])
        self._k_lin = float(beam_options['k_lin'])
        self._k_tor = float(beam_options['k_tor'])

        self.cross_section_fn = cross_section_fn
        self.attachment_fn = attachment_fn
        # Keys in support_dofs whose values are shaped (n_base, n_beams_per_coil);
        # _clamp_weights_for_spec slices [coil, local_beam] before passing to
        # attachment_fn so each call receives a scalar, not the full array.
        self._cross_section_dof_keys = tuple(cross_section_dof_keys)

        # Per-coil: which local base-coil index is the CC-beam end target,
        # and which symmetry transform to apply to its geometry.
        # 'none'   -> no transform (i < n_base - 1)
        # 'flip'   -> stellsym flip (i == n_base-1, stellsym=True)
        # 'rotate' -> rotate by 2π/nfp about z (i == n_base-1, stellsym=False)
        self._end_coil_local_idx, self._end_coil_transform = \
            self._build_end_coil_map()

        # Pre-build the static COO index arrays (I and J are loop-invariant).
        self._coo_I, self._coo_J = self._build_static_ij()

    # ============================================================================
    # Static topology helpers (called once at construction)
    # ============================================================================

    def _build_end_coil_map(self):
        """Build per-coil CC-beam endpoint target index and transform tag."""
        idx = []
        transform = []
        for i in range(self.n_base):
            if i < self.n_base - 1:
                idx.append(i + 1)
                transform.append('none')
            else:
                # Last coil wraps around
                if self.stellsym:
                    idx.append(self.n_base - 1)
                    transform.append('flip')
                else:
                    idx.append(0)
                    transform.append('rotate')
        return idx, transform

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
        return self._n_beam_cc 
        
    @property
    def n_beam_cf(self):
        return self._n_beam_cf
        
    @property
    def k_lin(self):
        return self._k_lin 
        
    @property
    def k_tor(self):
        return self._k_tor 

    @property
    def n_base(self):
        return len(self.base_curves_jax)
        
    @property
    def n_beams_per_coil(self):
        return self.n_beam_cc + self.n_beam_cf
        
    @property
    def n_beams_total(self):
        return self.n_base * self.n_beams_per_coil
        
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

    def _reconstruct_curves(
        self, base_curves_dofs: list[jax.Array]
    ) -> list[CurveXYZFourierJAX]:
        """Rebuild traced CurveXYZFourierJAX objects from current DOF arrays.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array]
            One DOF vector per base coil (traced quantities).

        Returns
        -------
        list[CurveXYZFourierJAX]
            Traced curve objects whose ``gamma_eval`` is differentiable
            w.r.t. each coil's DOFs.
        """
        return [
            CurveXYZFourierJAX(base.quadpoints, d, base.order)
            for base, d in zip(self.base_curves_jax, base_curves_dofs)
        ]

    def _apply_end_transform(
        self, pts: jax.Array, transform: str, inverse: bool = False
    ) -> jax.Array:
        """Apply (or invert) the wraparound symmetry transform.

        Parameters
        ----------
        pts : jax.Array, shape ``(..., 3)``
        transform : str
            One of ``'none'``, ``'flip'``, ``'rotate'``.
        inverse : bool
            When ``True``, apply the *inverse* of the named transform.
            ``'none'`` and ``'flip'`` are self-inverse; ``'rotate'`` uses
            ``−2π/nfp`` instead of ``+2π/nfp``.

        Returns
        -------
        jax.Array, same shape as ``pts``
        """
        if transform == 'none':
            return pts
        elif transform == 'flip':
            return _flip_points(pts)   # self-inverse
        else:  # 'rotate'
            angle = 2.0 * math.pi / self.nfp
            if inverse:
                angle = -angle
            return _rotate_points_z(pts, angle)

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
        phis_start_cc = support_dofs['phis_start_cc']       # (n_base, n_beam_cc)
        phis_end_cc   = support_dofs['phis_end_cc']         # (n_base, n_beam_cc)
        phis_start_cf = support_dofs['phis_start_cf']       # (n_base, n_beam_cf)
        x_foundation = support_dofs['x_foundation']       # (n_base, n_beam_cf, 3)

        x_start_list, x_end_list = [], []
        t_coil_start_list, t_coil_end_list = [], []

        for i, curve_i in enumerate(curves):
            end_idx = self._end_coil_local_idx[i]
            end_tfm = self._end_coil_transform[i]
            curve_end = curves[end_idx]

            # ── CC beams for coil i ──────────────────────────────────────────
            for j in range(self.n_beam_cc):
                phi_s = phis_start_cc[i, j]
                phi_e = phis_end_cc[i, j]

                x_s = curve_i.gamma_eval(phi_s)        # (3,)
                x_e_raw = curve_end.gamma_eval(phi_e)  # (3,) — may need transform
                x_e = self._apply_end_transform(x_e_raw, end_tfm)

                # Coil tangent vectors at the attachment phis
                t_cs_raw = curve_i.gamma_eval(phi_s, diff_order=1)         # (3,)
                t_cs = t_cs_raw / (jnp.linalg.norm(t_cs_raw) + 1e-300)
                t_ce_raw = curve_end.gamma_eval(phi_e, diff_order=1)       # (3,)
                t_ce_raw = self._apply_end_transform(t_ce_raw, end_tfm)
                t_ce = t_ce_raw / (jnp.linalg.norm(t_ce_raw) + 1e-300)

                x_start_list.append(x_s)
                x_end_list.append(x_e)
                t_coil_start_list.append(t_cs)
                t_coil_end_list.append(t_ce)

            # ── CF beams for coil i ──────────────────────────────────────────
            for j in range(self.n_beam_cf):
                phi_s = phis_start_cf[i, j]
                x_s = curve_i.gamma_eval(phi_s)        # (3,)
                x_e = x_foundation[i, j]               # (3,) — traced support DOF

                t_cs_raw = curve_i.gamma_eval(phi_s, diff_order=1)
                t_cs = t_cs_raw / (jnp.linalg.norm(t_cs_raw) + 1e-300)

                x_start_list.append(x_s)
                x_end_list.append(x_e)
                t_coil_start_list.append(t_cs)
                # CF: no coil tangent at foundation side → zero placeholder
                t_coil_end_list.append(jnp.zeros(3))

        x_start = jnp.stack(x_start_list, axis=0)         # (N, 3)
        x_end   = jnp.stack(x_end_list,   axis=0)         # (N, 3)
        t_coil_start = jnp.stack(t_coil_start_list, axis=0)  # (N, 3)
        t_coil_end   = jnp.stack(t_coil_end_list,   axis=0)  # (N, 3)

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
        theta_cc = support_dofs['thetas_orientation_cc']  # (n_base, n_beam_cc)
        theta_cf = support_dofs['thetas_orientation_cf']  # (n_base, n_beam_cf)

        t_beam = geom['t_beam']           # (N, 3)
        t_coil = geom['t_coil_start']     # (N, 3)

        # Flatten theta angles in the same beam ordering (cc first, then cf)
        theta_list = []
        for i in range(self.n_base):
            for j in range(self.n_beam_cc):
                theta_list.append(theta_cc[i, j])
            for j in range(self.n_beam_cf):
                theta_list.append(theta_cf[i, j])
        thetas = jnp.stack(theta_list, axis=0)  # (N,)

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
        * ``'x_ep'``      : ``(3,)`` endpoint position in the frame where
          ``coil`` surface points live (after symmetry transform when required).
        * ``'gamma3'``    : ``(3, 3)`` DCM for this beam
          (columns: ``x_local``, ``y_local``, ``z_local``).
        * ``'sign_x'``    : ``True`` at node-1 (beam extends toward
          ``+x_local``); ``False`` at node-2.
        * ``'tfm'``       : symmetry transform tag ``'none'`` / ``'flip'`` /
          ``'rotate'`` — applied to surface points before computing moment
          arms.  Always ``'none'`` for node-1.

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
        b = 0
        for i in range(self.n_base):
            end_idx = self._end_coil_local_idx[i]
            end_tfm = self._end_coil_transform[i]

            for j in range(self.n_beam_cc):
                g3 = gamma3[b]
                specs.append({
                    'b': b, 'node_side': 0, 'coil': i,
                    'x_ep': geom['x_start'][b], 'gamma3': g3,
                    'sign_x': True, 'tfm': 'none',
                })
                specs.append({
                    'b': b, 'node_side': 1, 'coil': end_idx,
                    'x_ep': geom['x_end'][b], 'gamma3': g3,
                    'sign_x': False, 'tfm': end_tfm,
                })
                b += 1

            for j in range(self.n_beam_cf):
                specs.append({
                    'b': b, 'node_side': 0, 'coil': i,
                    'x_ep': geom['x_start'][b], 'gamma3': gamma3[b],
                    'sign_x': True, 'tfm': 'none',
                })
                b += 1

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
        # Slice per-beam cross-section scalars out of (n_base, n_beams_per_coil) arrays
        # so attachment_fn receives the scalar for this specific beam.
        if self._cross_section_dof_keys:
            i = spec['coil']
            j = spec['b'] % self.n_beams_per_coil
            beam_dofs = {
                **support_dofs,
                **{k: support_dofs[k][i, j] for k in self._cross_section_dof_keys},
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
        """Rotate local stiffness to global frame: ``Γ_12^T K_local Γ_12``.

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
            return G12.T @ K_loc @ G12

        return jax.vmap(rotate_one)(K_local, Gamma_3)

    # ============================================================================
    # Support-ABC hook implementations
    # ============================================================================

    def compute_weights(
        self,
        coil_idx: int,
        surface_pts: jax.Array,
        curves_jax: list,
        dofs,
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

        Returns
        -------
        jax.Array, shape ``(n_surf,)``
        """
        if dofs is None:
            return Support.compute_weights(
                self, coil_idx, surface_pts, curves_jax, dofs
            )

        geom   = self._beam_geometry(curves_jax, dofs)
        gamma3 = self._direction_cosine_matrices(geom, dofs)
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

    def compute_attach(
        self,
        coil_idx: int,
        surface_pts: jax.Array,
        curves_jax: list,
        dofs,
        state: dict,
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

        Returns
        -------
        jax.Array, shape ``(n_surf, 3)``
        """
        u_s    = state['u_s']                            # (12 * N_beams,)
        u_beams = u_s.reshape(self.n_beams_total, 12)   # (N_beams, 12)
        n_surf  = surface_pts.shape[0]

        geom   = self._beam_geometry(curves_jax, dofs)
        gamma3 = self._direction_cosine_matrices(geom, dofs)
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

    def coupling_terms(
        self,
        base_curves_dofs,
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

        * Translation block: ``-k_{\\text{lin}} w_k^b I_3``
        * Rotation block:    ``+k_{\\text{lin}} w_k^b [r_k^b]×``
          (arising from ``θ × r = -[r]× θ``, sign folded)

        ``K_sc`` (row = beam endpoint DOF, col = coil translation DOF):

        * Beam translation, coil translation: ``-k_{\\text{lin}} w_k^b I_3``
        * Beam rotation, coil translation:    ``+k_{\\text{tor}} w_k^b [r_k^b]×``

        Parameters
        ----------
        base_curves_dofs : list[jax.Array]
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
        curves = self._reconstruct_curves(base_curves_dofs)
        geom   = self._beam_geometry(curves, support_dofs)
        gamma3 = self._direction_cosine_matrices(geom, support_dofs)
        specs  = self._endpoint_specs(geom, gamma3)

        I_cs_list, J_cs_list, V_cs_list = [], [], []
        I_sc_list, J_sc_list, V_sc_list = [], [], []

        for spec in specs:
            coil_i    = spec['coil']
            b         = spec['b']
            node_side = spec['node_side']

            surf_pts = surface_pts_by_coil[coil_i]
            surf_idx = surface_node_indices_by_coil[coil_i]
            n_surf   = surf_pts.shape[0]

            w_k, r_k = self._clamp_weights_for_spec(spec, surf_pts, support_dofs)

            t_off = 6 * node_side          # translation DOF within beam (0 or 6)
            r_off = 6 * node_side + 3      # rotation DOF within beam (3 or 9)

            beam_trans_global_base = support_dof_offset + 12 * b + t_off
            beam_rot_global_base   = support_dof_offset + 12 * b + r_off

            # ── K_cs: coil rows, beam cols ──────────────────────────────────
            for k in range(n_surf):
                node_global   = int(surf_idx[k])
                coil_dof_base = coil_dof_offsets[coil_i] + 3 * node_global
                w      = w_k[k]
                r      = r_k[k]
                skew_r = _skew(r)

                for d1 in range(3):
                    coil_dof = coil_dof_base + d1
                    I_cs_list.append(coil_dof)
                    J_cs_list.append(beam_trans_global_base + d1)
                    V_cs_list.append(-self.k_lin * w)
                    for d2 in range(3):
                        I_cs_list.append(coil_dof)
                        J_cs_list.append(beam_rot_global_base + d2)
                        V_cs_list.append(self.k_lin * w * skew_r[d1, d2])

            # ── K_sc: beam rows, coil cols ──────────────────────────────────
            for k in range(n_surf):
                node_global   = int(surf_idx[k])
                coil_dof_base = coil_dof_offsets[coil_i] + 3 * node_global
                w      = w_k[k]
                r      = r_k[k]
                skew_r = _skew(r)

                for d2 in range(3):
                    coil_dof       = coil_dof_base + d2
                    beam_trans_dof = beam_trans_global_base + d2
                    I_sc_list.append(beam_trans_dof)
                    J_sc_list.append(coil_dof)
                    V_sc_list.append(-self.k_lin * w)
                    for d1 in range(3):
                        I_sc_list.append(beam_rot_global_base + d1)
                        J_sc_list.append(coil_dof)
                        V_sc_list.append(self.k_tor * w * skew_r[d1, d2])

        if not I_cs_list:
            empty_i = jnp.zeros(0, dtype=jnp.int32)
            empty_v = jnp.zeros(0)
            return {
                'I_cs': empty_i, 'J_cs': empty_i, 'V_cs': empty_v,
                'I_sc': empty_i, 'J_sc': empty_i, 'V_sc': empty_v,
            }

        return {
            'I_cs': jnp.array(I_cs_list, dtype=jnp.int32),
            'J_cs': jnp.array(J_cs_list, dtype=jnp.int32),
            'V_cs': jnp.stack(V_cs_list),
            'I_sc': jnp.array(I_sc_list, dtype=jnp.int32),
            'J_sc': jnp.array(J_sc_list, dtype=jnp.int32),
            'V_sc': jnp.stack(V_sc_list),
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
            })

        # Append foundation entries for CF beams (node-2 side, no clamp)
        for i in range(self.n_base):
            for j in range(self.n_beam_cf):
                b      = i * self.n_beams_per_coil + self.n_beam_cc + j
                x_ep   = geom['x_end'][b]                  # = x_foundation[i, j]
                r_fnd  = x_foundation[i, j] - x_ep         # ~zero at rest
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

        For each endpoint ``e`` at node ``n`` (local DOFs ``6n:6n+3`` for
        translation, ``6n+3:6n+6`` for rotation):

        * **Translation–translation** (from linear spring,
          `F = k_lin * w * (u_endpoint - u_mesh)`)::

              K[3n:3n+3, 3n:3n+3] += (Σ_i w_i) * k_lin * I_3

        * **Torque–translation** (from `τ = k_tor * w * (u_endpoint - u_mesh) × r`,
          asymmetric)::

              K[3n+3:3n+6, 3n:3n+3] += k_tor * Σ_i (w_i * [r_i]×)

          NOTE: asymmetric K from user-specified torque law; K_ss is not
          symmetric.  A future cuDSS monolithic path must use mtype_id=0.

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
                else:
                    # Coil side: w is (n_surf,), r is (n_surf, 3)
                    w_sum = jnp.sum(w)         # scalar
                    # Weighted sum of skew matrices: Σ w_i [r_i]×
                    skew_sum = jnp.einsum('n,nij->ij', w,
                                          jax.vmap(_skew)(r))  # (3, 3)

                # Translation–translation: (Σ w_i) * k_lin * I_3
                K_tt = (self.k_lin * w_sum) * jnp.eye(3)
                K = K.at[t_off:t_off+3, t_off:t_off+3].add(K_tt)

                # Torque–translation: k_tor * Σ w_i [r_i]×  (asymmetric)
                K_rt = self.k_tor * skew_sum   # (3, 3)
                K = K.at[r_off:r_off+3, t_off:t_off+3].add(K_rt)

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
        base_curves_dofs: list[jax.Array],
        support_dofs: dict,
        surface_pts_by_coil: list[jax.Array] | None = None,
    ) -> tuple:
        """Return the support-local stiffness block ``K_ss`` in COO format.

        All outputs except ``I`` and ``J`` are differentiable w.r.t.
        ``base_curves_dofs`` and ``support_dofs``.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array]
            Traced coil DOF vectors, one per base coil.
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

        Returns
        -------
        I : np.ndarray, shape ``(N*144,)``  — static row indices
        J : np.ndarray, shape ``(N*144,)``  — static column indices
        V : jax.Array,  shape ``(N*144,)``  — stiffness values (traced)
        n_dofs : int                         — total support DOFs
        """
        # 1. Reconstruct traced curves from current coil DOFs.
        curves = self._reconstruct_curves(base_curves_dofs)

        # 2. Per-beam rest-state geometry (endpoint positions, lengths, tangents).
        geom = self._beam_geometry(curves, support_dofs)

        # 3. Cross-section properties from user callback.
        A_all, Iy_all, Iz_all, J_all = self.cross_section_fn(support_dofs)
        # Flatten from (n_base, n_beams_per_coil) to (N_beams,).
        A  = A_all.reshape(-1)
        Iy = Iy_all.reshape(-1)
        Iz = Iz_all.reshape(-1)
        Jj = J_all.reshape(-1)

        # 4. Bisymmetric beam stiffness in local frame (Eq. 4.34).
        K_local = self._local_stiffness(A, Iy, Iz, Jj, geom['L'])

        # 5. Direction-cosine matrices and rotation to global frame.
        Gamma_3  = self._direction_cosine_matrices(geom, support_dofs)
        K_global = self._global_stiffness(K_local, Gamma_3)

        # 6. Endpoint spring contributions (diagonal K_ss part).
        if surface_pts_by_coil is not None:
            beam_endpoints = self._endpoint_weights_and_r(
                curves, geom, Gamma_3, support_dofs, surface_pts_by_coil
            )
            K_spring = self._spring_stiffness_contributions(beam_endpoints)
            K_beam = K_global + K_spring
        else:
            K_beam = K_global  # bare-beam only; singular rank-6 blocks

        # 7. Scatter block-diagonal to flat COO triplets.
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
                    # Coil side: f += k_lin * Σ w_i u_mesh_i  (force target)
                    if u_mesh_by_coil is not None:
                        w          = ep['w']               # (n_surf,)
                        coil_for_ep = ep['coil']
                        u_mesh     = u_mesh_by_coil[coil_for_ep]   # (n_surf, 3)
                        f_trans    = self.k_lin * jnp.einsum('n,nd->d', w, u_mesh)
                        f = f.at[dof_base + t_off: dof_base + t_off + 3].add(f_trans)

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

            * ``'base_curves_dofs'`` : list[jax.Array]
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
        * ``'base_curves_dofs'``: echoed back for driver convenience.
        * ``'support_dofs'``    : echoed back for driver convenience.
        """
        base_curves_dofs    = inputs['base_curves_dofs']
        support_dofs        = inputs['support_dofs']
        surface_pts_by_coil = inputs['surface_pts_by_coil']
        u_mesh_by_coil      = inputs.get('u_mesh_by_coil', None)

        # Assemble K_ss (traced).
        I, J, V, n_dofs = self.coo(
            base_curves_dofs, support_dofs, surface_pts_by_coil
        )

        # Build dense K_ss from COO.  n_support_dofs is small in practice
        # (e.g. 5 coils × 6 beams × 12 DOFs = 360), so dense is fine.
        K_ss = jnp.zeros((n_dofs, n_dofs))
        K_ss = K_ss.at[I, J].add(V)

        # Assemble RHS.
        curves  = self._reconstruct_curves(base_curves_dofs)
        geom    = self._beam_geometry(curves, support_dofs)
        gamma3  = self._direction_cosine_matrices(geom, support_dofs)
        beam_endpoints = self._endpoint_weights_and_r(
            curves, geom, gamma3, support_dofs, surface_pts_by_coil
        )
        f_s = self._assemble_rhs(geom, beam_endpoints, u_mesh_by_coil)

        # Solve K_ss u_s = f_s using lineax (handles non-symmetric K).
        operator = lineax.MatrixLinearOperator(K_ss)
        solution = lineax.linear_solve(operator, f_s, solver=lineax.LU())
        u_s = solution.value

        return {
            'u_s':             u_s,
            'base_curves_dofs': base_curves_dofs,
            'support_dofs':    support_dofs,
        }
