"""Central support ring (CSR) beam-network coupling.

:class:`SupportBeamsCSR` extends :class:`~coil_fem.coupling.SupportBeams` with
a rectangular-section central support ring meshed over one field period and
coil-to-CSR (CR) beams that attach with an extra ``v_end_cr`` DOF.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy as onp

from ..geo import CurveRZFourierJAX, make_centroid_frame
from ..meshing import FramedCurveMeshRectangle
from ..pipelines import ElasticPipeline
from .beam_network import EndpointResult, EndpointSpec, SupportBeams, _rodrigues, _skew
from .supports import ContinuumMember


class SupportBeamsCSR(SupportBeams):
    """Beam network plus a one-field-period central support ring (CSR).

    The CSR is a :class:`~coil_fem.meshing.FramedCurveMeshRectangle` swept over
    ``phi ∈ [0, 1/nfp]`` with an exact periodic DOF tie at the seam.  Coil-to-
    CSR (CR) beams attach like CF beams topologically (one entry per base coil)
    but place their free end on the CSR surface with DOFs
    ``(phi_end_cr, v_end_cr)``.

    The ring lives entirely inside the support ``K_ss`` block, so
    :class:`~coil_fem.CoilFEM` needs no solve-path changes.

    Parameters
    ----------
    nfp, stellsym, beam_options, n_base, cross_section_fn, attachment_fn
        Forwarded to :class:`SupportBeams`.  ``beam_options`` must also contain
        ``n_beam_cr`` (int or length-``n_base`` sequence).
    csr_options : dict
        CSR mesh / material options.  Required keys: ``order``, ``w1``, ``w2``,
        ``n_phi``, ``E``, ``nu``.  Optional: ``n_grid_1``, ``n_grid_2``,
        ``mesh_type`` (default ``'TET4'``), ``n_quad`` (full-turn curve
        samples for the centroid frame; default ``max(64, 4*nfp*n_phi)``).
    problem_options : dict
        Same dict passed to :class:`~coil_fem.CoilFEM` so ``gpu_assembly``
        matches the coil pipelines.
    cross_section_dof_keys, fixed_clamp_fns, fixed_clamp_options
        Forwarded to :class:`SupportBeams`.

    Notes
    -----
    ``cross_section_fn`` must return per-group lists whose entry ``i < n_base``
    has shape ``(n_beam_cc[i] + n_beam_cf[i] + n_beam_cr[i],)`` (wrap group
    unchanged).  Beam assembly order is coil-major ``CC → CF → CR``, then the
    stellsym wrap group.
    """

    def __init__(
        self,
        nfp: int,
        stellsym: bool,
        beam_options: dict,
        n_base: int,
        cross_section_fn: Callable,
        attachment_fn: Callable,
        csr_options: dict,
        problem_options: dict,
        cross_section_dof_keys: tuple = (),
        fixed_clamp_fns=None,
        fixed_clamp_options=None,
    ):
        if 'n_beam_cr' not in beam_options:
            raise ValueError("beam_options must contain 'n_beam_cr'.")
        super().__init__(
            nfp=nfp,
            stellsym=stellsym,
            beam_options=beam_options,
            n_base=n_base,
            cross_section_fn=cross_section_fn,
            attachment_fn=attachment_fn,
            cross_section_dof_keys=cross_section_dof_keys,
            fixed_clamp_fns=fixed_clamp_fns,
            fixed_clamp_options=fixed_clamp_options,
        )

        self._n_beam_cr = self._check_beam_counts(
            beam_options['n_beam_cr'], n_base, 'n_beam_cr'
        )
        self._w_sym = 1.0 / (1 + int(stellsym))
        self._csr_options = dict(csr_options)
        self._problem_options = dict(problem_options)

        # Rebuild beam offsets: coil-major CC → CF → CR, then wrap.
        _per_coil = [
            self._n_beam_cc[i] + self._n_beam_cf[i] + self._n_beam_cr[i]
            for i in range(n_base)
        ]
        self._beam_offsets = tuple(int(sum(_per_coil[:i])) for i in range(n_base))
        self._wrap_beam_offset = int(sum(_per_coil))
        _n_wrap = self._n_beam_cc[n_base] if stellsym else 0
        self._n_beams_total = self._wrap_beam_offset + _n_wrap
        self._support_I, self._support_J = self._build_static_ij()
        self._csr_dof_offset = 12 * self._n_beams_total

        self._build_csr_mesh_and_pipeline()
        self._build_csr_reduction()

        # Re-JIT overrides (parent bound the pre-CR methods).
        cls = type(self)
        self.beam_geometry = jax.jit(cls.beam_geometry.__get__(self, cls))
        self.solve = jax.jit(cls.solve.__get__(self, cls))
        self.compute_weights = jax.jit(
            cls.compute_weights.__get__(self, cls), static_argnums=(0,),
        )

    # ========================================================================
    # Construction helpers
    # ========================================================================

    def _build_csr_mesh_and_pipeline(self) -> None:
        """Build the RZFourier sector mesh and its elastic pipeline."""
        opt = self._csr_options
        required = ('order', 'w1', 'w2', 'n_phi', 'E', 'nu')
        missing = [k for k in required if k not in opt]
        if missing:
            raise ValueError(f"csr_options missing required keys: {missing}.")

        order = int(opt['order'])
        n_phi = int(opt['n_phi'])
        n_quad = int(opt.get('n_quad', max(64, 4 * self._nfp * n_phi)))
        mesh_type = opt.get('mesh_type', 'TET4')
        w1, w2 = float(opt['w1']), float(opt['w2'])

        # Full-turn quadpoints so the centroid frame centre is the ring axis.
        qp = jnp.linspace(0.0, 1.0, n_quad, endpoint=False)
        if self._stellsym:
            n_dofs = 2 * order + 1
        else:
            n_dofs = 4 * order + 2
        # Seed: circular ring of radius 1 (rc_0 = 1); caller overwrites via
        # support_dofs['csr_curve_dofs'].
        dofs0 = jnp.zeros(n_dofs)
        dofs0 = dofs0.at[0].set(1.0)

        self._csr_curve_template = CurveRZFourierJAX(
            qp, dofs0, order, self._nfp, self._stellsym,
        )
        fc = make_centroid_frame(self._csr_curve_template)
        self.csr_mesh = FramedCurveMeshRectangle(
            fc, w1, w2,
            n_grid_1=opt.get('n_grid_1'),
            n_grid_2=opt.get('n_grid_2'),
            mesh_type=mesh_type,
            phi_span=1.0 / self._nfp,
            n_phi=n_phi,
        )
        self._csr_framed_template = self.csr_mesh.framed_curve
        self._csr_a = w1
        self._csr_b = w2

        # assemble_coo requires gpu_assembly; the CSR never calls fwd_pred, so
        # forcing solver='cudss' here only selects the device-side Jacobian path
        # and does not require a CUDA device at construction time.
        csr_po = {**self._problem_options, 'solver': 'cudss'}
        self._csr_pipeline = ElasticPipeline(
            self.csr_mesh,
            float(opt['E']),
            float(opt['nu']),
            None,
            (0.0, 0.0, 0.0),
            csr_po,
        )

    def _build_csr_reduction(self) -> None:
        """Build static seam-tie arrays for the one-period CSR mesh."""
        mesh = self.csr_mesh
        n_nodes = int(mesh.points.shape[0])
        u = mesh.u_per_node
        v = mesh.v_per_node
        phi_idx = mesh.phi_idx_per_node
        phi_max = int(phi_idx.max())

        near = np.where(phi_idx == 0)[0]
        far = np.where(phi_idx == phi_max)[0]
        if near.shape[0] != far.shape[0]:
            raise RuntimeError(
                f"CSR seam faces differ in size: near={near.shape[0]}, "
                f"far={far.shape[0]}."
            )

        # Pair far → near by rounded (u, v) keys.
        def _key(i):
            return (round(float(u[i]), 10), round(float(v[i]), 10))

        near_map = {_key(i): int(i) for i in near}
        if len(near_map) != near.shape[0]:
            raise RuntimeError(
                "CSR near-face (u, v) keys are not unique; cannot pair seam."
            )
        far_to_near = np.empty(n_nodes, dtype=np.int32)
        far_to_near[:] = -1
        for i in far:
            k = _key(i)
            if k not in near_map:
                raise RuntimeError(
                    f"CSR far-face node {i} has no near-face partner at {k}."
                )
            far_to_near[i] = near_map[k]
        if len({int(far_to_near[i]) for i in far}) != far.shape[0]:
            raise RuntimeError("CSR far→near pairing is not a bijection.")

        is_slave = np.zeros(n_nodes, dtype=bool)
        is_slave[far] = True
        # Compact reduced index for free (non-slave) nodes.
        free = np.where(~is_slave)[0]
        red_of_free = -np.ones(n_nodes, dtype=np.int32)
        red_of_free[free] = np.arange(free.shape[0], dtype=np.int32)
        red_node = red_of_free.copy()
        red_node[far] = red_of_free[far_to_near[far]]

        # node_Q[i] = I for masters, R_z(2π/nfp) for slaves (u_far = Q u_near).
        c = math.cos(2.0 * math.pi / self._nfp)
        s = math.sin(2.0 * math.pi / self._nfp)
        Q_period = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64,
        )
        node_Q = np.repeat(np.eye(3, dtype=np.float64)[None], n_nodes, axis=0)
        node_Q[far] = Q_period

        self._csr_red_node = red_node
        self._csr_is_slave = is_slave
        self._csr_node_Q = node_Q
        self._csr_Q_period = Q_period
        self._n_csr_dofs = 3 * int(free.shape[0])
        self._csr_n_free_nodes = int(free.shape[0])

        # End-cap mask: zero Winkler on the two cut faces.
        prob = self._csr_pipeline.problem
        bi = np.asarray(prob.boundary_inds_list[0], dtype=np.int32)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        n_sel = bi.shape[0]
        n_fq = int(prob._sel_face_sv.shape[1])
        # Face-local global nodes from the winkler maps.
        face_to_node = np.asarray(prob._surf_face_to_surf_node)  # (n_sel, n_fn)
        unique_glob = np.asarray(prob._surf_unique_global_nodes)
        endcap = np.ones((n_sel, n_fq), dtype=np.float64)
        for s in range(n_sel):
            gnodes = unique_glob[face_to_node[s]]
            pidx = phi_idx[gnodes]
            if np.all((pidx == 0) | (pidx == phi_max)):
                endcap[s, :] = 0.0
        self._csr_endcap_mask = endcap
        self._csr_endcap_mask_flat = endcap.reshape(-1)

        # Reduce the CSR volume+surface COO pattern.
        I_full = np.asarray(prob.I, dtype=np.int32)
        J_full = np.asarray(prob.J, dtype=np.int32)
        self._reduce_csr_coo_pattern(I_full, J_full)

        # Pre-build beam↔CSR coupling pattern (filled after geom is known at
        # pattern time via surface node indices).
        self._csr_beam_spring_I = None
        self._csr_beam_spring_J = None
        self._build_csr_beam_spring_pattern()

    def _reduce_csr_coo_pattern(self, I_full, J_full) -> None:
        """Map full-mesh COO (I, J) to reduced seam-tied DOFs."""
        red_node = self._csr_red_node
        node_Q = self._csr_node_Q
        is_slave = self._csr_is_slave

        node_i = I_full // 3
        comp_i = I_full % 3
        node_j = J_full // 3
        comp_j = J_full % 3

        touches_slave = is_slave[node_i] | is_slave[node_j]
        plain = np.where(~touches_slave)[0]
        exp = np.where(touches_slave)[0]

        # Plain entries: identity map into reduced DOFs.
        I_plain = (3 * red_node[node_i[plain]] + comp_i[plain]).astype(np.int32)
        J_plain = (3 * red_node[node_j[plain]] + comp_j[plain]).astype(np.int32)

        # Expanding entries: V * outer(Q_i[row], Q_j[col]) → 9 reduced slots.
        n_exp = exp.shape[0]
        Qi = node_Q[node_i[exp]]  # (n_exp, 3, 3)
        Qj = node_Q[node_j[exp]]
        # For entry (node_p, a), (node_q, b): weight[a', b'] = Qi[a, a'] * Qj[b, b']
        # I_red = 3*red[p] + a', J_red = 3*red[q] + b'
        a = comp_i[exp]
        b = comp_j[exp]
        # Gather rows of Qi / Qj for the original component.
        Qi_row = Qi[np.arange(n_exp), a, :]  # (n_exp, 3)
        Qj_row = Qj[np.arange(n_exp), b, :]  # (n_exp, 3)
        W = Qi_row[:, :, None] * Qj_row[:, None, :]  # (n_exp, 3, 3)

        red_i = red_node[node_i[exp]]
        red_j = red_node[node_j[exp]]
        ap, bp = np.meshgrid(np.arange(3), np.arange(3), indexing='ij')
        I_exp = (3 * red_i[:, None, None] + ap[None, :, :]).reshape(-1).astype(np.int32)
        J_exp = (3 * red_j[:, None, None] + bp[None, :, :]).reshape(-1).astype(np.int32)

        self._csr_I = np.concatenate([I_plain, I_exp]).astype(np.int32)
        self._csr_J = np.concatenate([J_plain, J_exp]).astype(np.int32)
        self._csr_plain_idx = plain.astype(np.int32)
        self._csr_exp_idx = exp.astype(np.int32)
        self._csr_W = W.astype(np.float64)

    def _build_csr_beam_spring_pattern(self) -> None:
        """Static I/J for beam↔CSR spring blocks (both orientations)."""
        # Pattern depends only on beam indices and CSR surface node set.
        surf_glob = np.asarray(
            self._csr_pipeline.problem._surf_unique_global_nodes, dtype=np.int32,
        )
        n_surf = surf_glob.shape[0]
        red = self._csr_red_node[surf_glob]  # (n_surf,) reduced node ids
        d3 = np.arange(3, dtype=np.int32)
        csr_base = self._csr_dof_offset

        I_parts, J_parts = [], []

        def _add(b, node_side):
            t_off = 6 * node_side
            r_off = 6 * node_side + 3
            beam_t = (12 * b + t_off + d3).astype(np.int32)
            beam_r = (12 * b + r_off + d3).astype(np.int32)
            csr_dofs = (csr_base + 3 * red[:, None] + d3[None, :]).astype(np.int32)

            # Beam rows ↔ CSR cols (and transpose).
            # K_bs: rows beam, cols csr — 2 blocks (trans, rot) × n_surf × 9
            rows_bt = np.broadcast_to(beam_t[None, :, None], (n_surf, 3, 3))
            rows_br = np.broadcast_to(beam_r[None, :, None], (n_surf, 3, 3))
            cols_c = np.broadcast_to(csr_dofs[:, None, :], (n_surf, 3, 3))
            I_parts.append(rows_bt.reshape(-1).copy())
            I_parts.append(rows_br.reshape(-1).copy())
            J_parts.append(cols_c.reshape(-1).copy())
            J_parts.append(cols_c.reshape(-1).copy())

            # K_sb: rows csr, cols beam
            rows_c = np.broadcast_to(csr_dofs[:, :, None], (n_surf, 3, 3))
            cols_bt = np.broadcast_to(beam_t[None, None, :], (n_surf, 3, 3))
            cols_br = np.broadcast_to(beam_r[None, None, :], (n_surf, 3, 3))
            I_parts.append(rows_c.reshape(-1).copy())
            I_parts.append(rows_c.reshape(-1).copy())
            J_parts.append(cols_bt.reshape(-1).copy())
            J_parts.append(cols_br.reshape(-1).copy())

        for i in range(self.n_base):
            for j in range(self._n_beam_cr[i]):
                b = (
                    self._beam_offsets[i]
                    + self.n_beam_cc[i]
                    + self.n_beam_cf[i]
                    + j
                )
                # Coil side already in K_cs; CSR side is node_side=1.
                _add(b, 1)
                if self.stellsym:
                    # Flip-half image shares the same beam DOFs.
                    _add(b, 1)

        if not I_parts:
            empty = np.zeros(0, dtype=np.int32)
            self._csr_beam_spring_I = empty
            self._csr_beam_spring_J = empty
            self._n_csr_beam_spring_endpoints = 0
        else:
            self._csr_beam_spring_I = np.concatenate(I_parts).astype(np.int32)
            self._csr_beam_spring_J = np.concatenate(J_parts).astype(np.int32)
            self._n_csr_beam_spring_endpoints = (
                sum(self._n_beam_cr) * (2 if self.stellsym else 1)
            )

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def n_beam_cr(self):
        """Per-coil coil-to-CSR beam counts, length-``n_base`` tuple of int."""
        return self._n_beam_cr

    @property
    def n_beams_per_coil(self):
        return tuple(
            self._n_beam_cc[i] + self._n_beam_cf[i] + self._n_beam_cr[i]
            for i in range(self.n_base)
        )

    @property
    def n_support_dofs(self):
        return 12 * self.n_beams_total + self._n_csr_dofs

    @property
    def csr_dof_offset(self) -> int:
        """Support-local index where CSR reduced DOFs begin."""
        return self._csr_dof_offset

    # ========================================================================
    # Geometry
    # ========================================================================

    def beam_geometry(self, curves_jax: list, support_dofs: dict) -> dict:
        """Per-beam geometry including CR beams and CSR mesh points."""
        phis_start_cc = support_dofs['phis_start_cc']
        phis_end_cc = support_dofs['phis_end_cc']
        phis_start_cf = support_dofs['phis_start_cf']
        x_foundation = support_dofs['x_foundation']
        phis_start_cr = support_dofs['phis_start_cr']
        phis_end_cr = support_dofs['phis_end_cr']
        v_end_cr = support_dofs['v_end_cr']

        csr_dofs = support_dofs['csr_curve_dofs']
        tmpl = self._csr_curve_template
        csr_curve = CurveRZFourierJAX(
            tmpl.quadpoints, csr_dofs, tmpl.order, tmpl.nfp, tmpl.stellsym,
        )
        csr_fc = self._csr_framed_template.with_dofs(csr_dofs)
        csr_points = self.csr_mesh.mesh_points_from_dofs(csr_dofs)

        x_start_list, x_end_list = [], []
        t_coil_start_list, t_coil_end_list = [], []

        def append_cc_group(g):
            n_g = self.n_beam_cc[g]
            if n_g == 0:
                return
            start_idx, end_idx, end_tfm = self._cc_groups[g]
            curve_s = curves_jax[start_idx]
            curve_e = curves_jax[end_idx]
            phi_s_g = phis_start_cc[g]
            phi_e_g = phis_end_cc[g]

            x_s_g = curve_s.gamma_eval(phi_s_g)
            x_e_raw = curve_e.gamma_eval(phi_e_g)
            x_e_g = self._apply_end_transform(x_e_raw, end_tfm)

            t_cs_raw = curve_s.gamma_eval(phi_s_g, diff_order=1)
            t_cs_g = t_cs_raw / (
                jnp.linalg.norm(t_cs_raw, axis=1, keepdims=True) + 1e-300
            )
            t_ce_raw = curve_e.gamma_eval(phi_e_g, diff_order=1)
            t_ce_raw = self._apply_end_transform(t_ce_raw, end_tfm)
            t_ce_g = t_ce_raw / (
                jnp.linalg.norm(t_ce_raw, axis=1, keepdims=True) + 1e-300
            )

            x_start_list.append(x_s_g)
            x_end_list.append(x_e_g)
            t_coil_start_list.append(t_cs_g)
            t_coil_end_list.append(t_ce_g)

        for i, curve_i in enumerate(curves_jax):
            append_cc_group(i)

            n_cf_i = self.n_beam_cf[i]
            if n_cf_i > 0:
                phi_s_cf = phis_start_cf[i]
                x_s_cf = curve_i.gamma_eval(phi_s_cf)
                t_cs_raw = curve_i.gamma_eval(phi_s_cf, diff_order=1)
                t_cs_cf = t_cs_raw / (
                    jnp.linalg.norm(t_cs_raw, axis=1, keepdims=True) + 1e-300
                )
                x_start_list.append(x_s_cf)
                x_end_list.append(x_foundation[i])
                t_coil_start_list.append(t_cs_cf)
                t_coil_end_list.append(jnp.zeros((n_cf_i, 3)))

            n_cr_i = self._n_beam_cr[i]
            if n_cr_i > 0:
                phi_s = phis_start_cr[i]
                phi_e = phis_end_cr[i]
                v_e = v_end_cr[i]
                x_s = curve_i.gamma_eval(phi_s)
                t_cs_raw = curve_i.gamma_eval(phi_s, diff_order=1)
                t_cs = t_cs_raw / (
                    jnp.linalg.norm(t_cs_raw, axis=1, keepdims=True) + 1e-300
                )
                gamma_e = csr_curve.gamma_eval(phi_e)
                _, _, q_e = csr_fc.rotated_frame_eval(phi_e)
                x_e = gamma_e + (self._csr_b / 2.0) * v_e[:, None] * q_e
                t_ce_raw = csr_curve.gamma_eval(phi_e, diff_order=1)
                t_ce = t_ce_raw / (
                    jnp.linalg.norm(t_ce_raw, axis=1, keepdims=True) + 1e-300
                )
                x_start_list.append(x_s)
                x_end_list.append(x_e)
                t_coil_start_list.append(t_cs)
                t_coil_end_list.append(t_ce)

        if self.stellsym:
            append_cc_group(self.n_base)

        x_start = jnp.concatenate(x_start_list, axis=0)
        x_end = jnp.concatenate(x_end_list, axis=0)
        t_coil_start = jnp.concatenate(t_coil_start_list, axis=0)
        t_coil_end = jnp.concatenate(t_coil_end_list, axis=0)

        diff = x_end - x_start
        L = jnp.linalg.norm(diff, axis=1)
        t_beam = diff / (L[:, None] + 1e-300)

        theta_cc = support_dofs['thetas_orientation_cc']
        theta_cf = support_dofs['thetas_orientation_cf']
        theta_cr = support_dofs['thetas_orientation_cr']
        theta_parts = []
        for i in range(self.n_base):
            if self.n_beam_cc[i] > 0:
                theta_parts.append(theta_cc[i])
            if self.n_beam_cf[i] > 0:
                theta_parts.append(theta_cf[i])
            if self._n_beam_cr[i] > 0:
                theta_parts.append(theta_cr[i])
        if self.stellsym and self.n_beam_cc[self.n_base] > 0:
            theta_parts.append(theta_cc[self.n_base])
        thetas = jnp.concatenate(theta_parts, axis=0)

        def single_dcm(t_b, t_c, theta):
            ref = jnp.cross(t_b, t_c)
            ref_norm = jnp.linalg.norm(ref)
            ref = jnp.where(
                ref_norm > 1e-9,
                ref / ref_norm,
                jnp.array([0., 0., 1.]) - t_b * t_b[2],
            )
            ref = ref / (jnp.linalg.norm(ref) + 1e-300)
            z_local = _rodrigues(t_b, 2.0 * jnp.pi * theta) @ ref
            z_local = z_local / (jnp.linalg.norm(z_local) + 1e-300)
            x_local = t_b
            y_local = jnp.cross(z_local, x_local)
            y_local = y_local / (jnp.linalg.norm(y_local) + 1e-300)
            return jnp.stack([x_local, y_local, z_local], axis=1)

        xi_start, xi_end, L_eff = self._surface_exit_params(
            curves_jax, support_dofs, x_start, x_end, L,
        )
        return {
            'x_start': x_start,
            'x_end': x_end,
            't_beam': t_beam,
            'L': L,
            't_coil_start': t_coil_start,
            't_coil_end': t_coil_end,
            'gamma3': jax.vmap(single_dcm)(t_beam, t_coil_start, thetas),
            'xi_start': xi_start,
            'xi_end': xi_end,
            'L_eff': L_eff,
            'csr_points': csr_points,
        }

    def _surface_exit_params(
        self,
        curves_jax: list,
        support_dofs: dict,
        x_start: jax.Array,
        x_end: jax.Array,
        L: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """``xi_start`` / ``xi_end`` / ``L_eff`` including CR beams."""
        n = L.shape[0]
        if self._coil_framed_templates is None:
            return jnp.zeros(n), jnp.ones(n), L

        fcs = [
            tmpl.with_dofs(c.dofs)
            for tmpl, c in zip(self._coil_framed_templates, curves_jax)
        ]
        is_disk = jnp.asarray(self._coil_is_disk)
        a_all = jnp.asarray(self._coil_a)
        b_all = jnp.asarray(self._coil_b)

        phis_start_cc = support_dofs['phis_start_cc']
        phis_end_cc = support_dofs['phis_end_cc']
        phis_start_cf = support_dofs['phis_start_cf']
        phis_start_cr = support_dofs['phis_start_cr']
        phis_end_cr = support_dofs['phis_end_cr']

        csr_dofs = support_dofs['csr_curve_dofs']
        csr_fc = self._csr_framed_template.with_dofs(csr_dofs)

        xi_s_list, xi_e_list = [], []
        b0 = 0

        def append_cc_group(g):
            nonlocal b0
            n_g = self.n_beam_cc[g]
            if n_g == 0:
                return
            start_idx, end_idx, end_tfm = self._cc_groups[g]
            sl = slice(b0, b0 + n_g)
            d = x_end[sl] - x_start[sl]
            _, p_s, q_s = fcs[start_idx].rotated_frame_eval(phis_start_cc[g])
            _, p_e, q_e = fcs[end_idx].rotated_frame_eval(phis_end_cc[g])
            p_e = self._apply_end_transform(p_e, end_tfm)
            q_e = self._apply_end_transform(q_e, end_tfm)
            xi_s_list.append(self._xi_surface_exit(
                d, p_s, q_s, a_all[start_idx], b_all[start_idx], is_disk[start_idx],
            ))
            xi_e_list.append(
                1.0 - self._xi_surface_exit(
                    d, p_e, q_e, a_all[end_idx], b_all[end_idx], is_disk[end_idx],
                )
            )
            b0 += n_g

        for i in range(self.n_base):
            append_cc_group(i)
            n_cf = self.n_beam_cf[i]
            if n_cf > 0:
                sl = slice(b0, b0 + n_cf)
                d = x_end[sl] - x_start[sl]
                _, p_s, q_s = fcs[i].rotated_frame_eval(phis_start_cf[i])
                xi_s_list.append(self._xi_surface_exit(
                    d, p_s, q_s, a_all[i], b_all[i], is_disk[i],
                ))
                xi_e_list.append(jnp.ones(n_cf))
                b0 += n_cf

            n_cr = self._n_beam_cr[i]
            if n_cr > 0:
                sl = slice(b0, b0 + n_cr)
                d = x_end[sl] - x_start[sl]
                _, p_s, q_s = fcs[i].rotated_frame_eval(phis_start_cr[i])
                _, p_e, q_e = csr_fc.rotated_frame_eval(phis_end_cr[i])
                xi_s_list.append(self._xi_surface_exit(
                    d, p_s, q_s, a_all[i], b_all[i], is_disk[i],
                ))
                xi_e_list.append(
                    1.0 - self._xi_surface_exit(
                        d, p_e, q_e,
                        jnp.asarray(self._csr_a),
                        jnp.asarray(self._csr_b),
                        jnp.asarray(False),
                    )
                )
                b0 += n_cr

        if self.stellsym:
            append_cc_group(self.n_base)

        xi_start = jnp.clip(jnp.concatenate(xi_s_list, axis=0), 0.0, 1.0)
        xi_end = jnp.clip(jnp.concatenate(xi_e_list, axis=0), 0.0, 1.0)
        L_eff = jnp.maximum(L * (xi_end - xi_start), 1e-3 * L)
        return xi_start, xi_end, L_eff

    # ========================================================================
    # Endpoints
    # ========================================================================

    def _endpoint_specs(self, geom: dict, gamma3: jax.Array):
        """Coil-coupled endpoints including CR coil-side attachments."""
        specs: list[EndpointSpec] = []

        def append_cc_group(g, b):
            start_idx, end_idx, end_tfm = self._cc_groups[g]
            for j in range(self.n_beam_cc[g]):
                g3 = gamma3[b]
                specs.append(EndpointSpec(
                    b=b, coil_origin=g, j_local=j,
                    node_side=0, coil=start_idx,
                    x_ep=geom['x_start'][b], gamma3=g3,
                    sign_x=True, tfm='none',
                ))
                specs.append(EndpointSpec(
                    b=b, coil_origin=g, j_local=j,
                    node_side=1, coil=end_idx,
                    x_ep=geom['x_end'][b], gamma3=g3,
                    sign_x=False, tfm=end_tfm,
                ))
                b += 1
            return b

        b = 0
        for i in range(self.n_base):
            b = append_cc_group(i, b)
            for j in range(self.n_beam_cf[i]):
                j_local = self.n_beam_cc[i] + j
                specs.append(EndpointSpec(
                    b=b, coil_origin=i, j_local=j_local,
                    node_side=0, coil=i,
                    x_ep=geom['x_start'][b], gamma3=gamma3[b],
                    sign_x=True, tfm='none',
                ))
                b += 1
            for j in range(self._n_beam_cr[i]):
                j_local = self.n_beam_cc[i] + self.n_beam_cf[i] + j
                specs.append(EndpointSpec(
                    b=b, coil_origin=i, j_local=j_local,
                    node_side=0, coil=i,
                    x_ep=geom['x_start'][b], gamma3=gamma3[b],
                    sign_x=True, tfm='none',
                ))
                b += 1

        if self.stellsym:
            append_cc_group(self.n_base, b)

        specs_by_coil: dict[int, list[EndpointSpec]] = {}
        for spec in specs:
            specs_by_coil.setdefault(spec.coil, []).append(spec)
        return specs, specs_by_coil

    def _csr_endpoint_specs(self, geom: dict) -> list[EndpointSpec]:
        """Ring-side CR endpoints (``coil=-2``); plus flip_half images."""
        gamma3 = geom['gamma3']
        specs: list[EndpointSpec] = []
        for i in range(self.n_base):
            for j in range(self._n_beam_cr[i]):
                b = (
                    self._beam_offsets[i]
                    + self.n_beam_cc[i]
                    + self.n_beam_cf[i]
                    + j
                )
                j_local = self.n_beam_cc[i] + self.n_beam_cf[i] + j
                specs.append(EndpointSpec(
                    b=b, coil_origin=i, j_local=j_local,
                    node_side=1, coil=-2,
                    x_ep=geom['x_end'][b], gamma3=gamma3[b],
                    sign_x=False, tfm='none',
                ))
                if self.stellsym:
                    specs.append(EndpointSpec(
                        b=b, coil_origin=i, j_local=j_local,
                        node_side=1, coil=-2,
                        x_ep=geom['x_end'][b], gamma3=gamma3[b],
                        sign_x=False, tfm='flip_half',
                    ))
        return specs

    def _endpoint_weights_and_r(
        self,
        curves,
        geom: dict,
        gamma3: jax.Array,
        support_dofs: dict,
        surface_pts_by_coil: list[jax.Array],
        jxw_by_coil: list | None = None,
    ) -> list[list[EndpointResult]]:
        """Coil-side endpoints plus ring-side springs (``w_sym``-scaled jxw)."""
        beam_eps = SupportBeams._endpoint_weights_and_r(
            self, curves, geom, gamma3, support_dofs,
            surface_pts_by_coil, jxw_by_coil,
        )

        csr_pts = geom['csr_points']
        csr_surf = self._csr_pipeline.surface_quad_points(csr_pts)
        csr_jxw = self._csr_pipeline.problem.surface_jxw(csr_pts).reshape(-1)
        # Kill fictitious cut-face quads.
        csr_jxw = csr_jxw * jnp.asarray(self._csr_endcap_mask_flat)
        w_sym = self._w_sym

        for spec in self._csr_endpoint_specs(geom):
            w_k, r_k = self._clamp_weights_for_spec(spec, csr_surf, support_dofs)
            beam_eps[spec.b].append(EndpointResult(
                w=w_k, r=r_k,
                jxw=csr_jxw * w_sym,
                node_side=spec.node_side, coil=spec.coil, tfm=spec.tfm,
            ))
        return beam_eps

    # ========================================================================
    # Coupling / support blocks
    # ========================================================================

    def coupling_pattern(
        self,
        coil_dof_offsets: list[int],
        support_dof_offset: int,
        surface_node_indices_by_coil: list,
    ) -> tuple:
        """K_cs / K_sc pattern including CR coil-side endpoints."""
        d3 = np.arange(3, dtype=np.int32)
        I_cs_parts, J_cs_parts = [], []
        I_sc_parts, J_sc_parts = [], []

        def _add_endpoint(b: int, node_side: int, coil_i: int) -> None:
            surf_idx = np.asarray(
                surface_node_indices_by_coil[coil_i], dtype=np.int32,
            )
            n_surf = surf_idx.shape[0]
            t_off = 6 * node_side
            r_off = 6 * node_side + 3
            beam_trans_base = support_dof_offset + 12 * b + t_off
            beam_rot_base = support_dof_offset + 12 * b + r_off
            coil_dof_base = coil_dof_offsets[coil_i] + 3 * surf_idx
            coil_dofs = coil_dof_base[:, None] + d3[None, :]
            beam_trans_dofs = (beam_trans_base + d3).astype(np.int32)
            beam_rot_dofs = (beam_rot_base + d3).astype(np.int32)

            rows_cs = np.broadcast_to(coil_dofs[:, :, None], (n_surf, 3, 3))
            cols_cs_t = np.broadcast_to(
                beam_trans_dofs[None, None, :], (n_surf, 3, 3),
            )
            cols_cs_r = np.broadcast_to(
                beam_rot_dofs[None, None, :], (n_surf, 3, 3),
            )
            I_cs_parts.append(rows_cs.reshape(-1).copy())
            I_cs_parts.append(rows_cs.reshape(-1).copy())
            J_cs_parts.append(cols_cs_t.reshape(-1).copy())
            J_cs_parts.append(cols_cs_r.reshape(-1).copy())

            rows_sc_t = np.broadcast_to(
                beam_trans_dofs[None, :, None], (n_surf, 3, 3),
            )
            rows_sc_r = np.broadcast_to(
                beam_rot_dofs[None, :, None], (n_surf, 3, 3),
            )
            cols_sc = np.broadcast_to(coil_dofs[:, None, :], (n_surf, 3, 3))
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
            for _j in range(self._n_beam_cr[i]):
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

    def support_pattern(self):
        """Local COO I/J for beams + reduced CSR + beam↔CSR springs."""
        I_beam, J_beam = self._support_I, self._support_J
        off = self._csr_dof_offset
        I_csr = self._csr_I + off
        J_csr = self._csr_J + off
        I_spr = self._csr_beam_spring_I
        J_spr = self._csr_beam_spring_J
        return (
            np.concatenate([I_beam, I_csr, I_spr]).astype(np.int32),
            np.concatenate([J_beam, J_csr, J_spr]).astype(np.int32),
        )

    def support_values(
        self,
        curves_jax: list,
        support_dofs: dict,
        surface_pts_by_coil: list[jax.Array] | None = None,
        geom: dict | None = None,
        *,
        jxw_by_coil: list | None = None,
        beam_endpoints: list[list[EndpointResult]] | None = None,
    ) -> jax.Array:
        """COO values: beams + reduced CSR stiffness + beam↔CSR springs."""
        if geom is None:
            geom = self.beam_geometry(curves_jax, support_dofs)

        V_beams = SupportBeams.support_values(
            self, curves_jax, support_dofs, surface_pts_by_coil, geom,
            jxw_by_coil=jxw_by_coil, beam_endpoints=beam_endpoints,
        )

        # ── Ring block ──────────────────────────────────────────────────────
        csr_pts = geom['csr_points']
        csr_surf = self._csr_pipeline.surface_quad_points(csr_pts)
        # Attachment weights on the ring surface.
        w_a = jnp.zeros(csr_surf.shape[0])
        for spec in self._csr_endpoint_specs(geom):
            w_k, _ = self._clamp_weights_for_spec(spec, csr_surf, support_dofs)
            w_a = w_a + w_k
        w_g = jnp.zeros_like(w_a)
        support_k = (
            (self.k_attachment * w_a + self.k_clamp * w_g)
            * jnp.asarray(self._csr_endcap_mask_flat)
        )
        n_cells = self._csr_pipeline.problem.fes[0].num_cells
        n_quads = self._csr_pipeline.problem.fes[0].num_quads
        bf0 = jnp.zeros((n_cells, n_quads, 3))
        _, _, V_full, _, _ = self._csr_pipeline.assemble_coo({
            'points': csr_pts,
            'body_force': bf0,
            'support_k': support_k,
        })
        V_full = jnp.asarray(V_full)
        V_plain = V_full[self._csr_plain_idx]
        V_exp = (
            V_full[self._csr_exp_idx][:, None, None]
            * jnp.asarray(self._csr_W)
        ).reshape(-1)
        V_csr = jnp.concatenate([V_plain, V_exp]) * self._w_sym

        # ── Beam↔CSR off-diagonal springs ───────────────────────────────────
        V_spr = self._csr_beam_spring_values(geom, support_dofs, csr_surf)

        return jnp.concatenate([V_beams, V_csr, V_spr])

    def _csr_beam_spring_values(
        self,
        geom: dict,
        support_dofs: dict,
        csr_surf: jax.Array,
    ) -> jax.Array:
        """Traced V for beam↔CSR spring blocks (aligned with pattern)."""
        if self._n_csr_beam_spring_endpoints == 0:
            return jnp.zeros(0)

        prob = self._csr_pipeline.problem
        face_sv = prob._sel_face_sv
        face_to_surf = prob._surf_face_to_surf_node
        n_surf_nodes = int(prob._surf_unique_global_nodes.shape[0])
        surf_glob = np.asarray(prob._surf_unique_global_nodes, dtype=np.int32)
        node_Q = jnp.asarray(self._csr_node_Q)  # (n_nodes, 3, 3)
        Q_nodes = node_Q[surf_glob]  # (n_surf, 3, 3)

        csr_jxw = prob.surface_jxw(geom['csr_points'])  # (n_sel, n_fq)
        csr_jxw = csr_jxw * jnp.asarray(self._csr_endcap_mask)
        w_sym = self._w_sym
        k = self.k_attachment

        V_parts: list = []
        for spec in self._csr_endpoint_specs(geom):
            w_k, r_k = self._clamp_weights_for_spec(spec, csr_surf, support_dofs)
            Q = jnp.asarray(self._tfm_Q[spec.tfm])
            Qinv = Q.T

            n_sel = face_sv.shape[0]
            n_fq = face_sv.shape[1]
            w_sq = w_k.reshape(n_sel, n_fq)
            r_sq = r_k.reshape(n_sel, n_fq, 3)
            skew_sq = jax.vmap(jax.vmap(_skew))(r_sq)
            wjxw_sq = w_sq * csr_jxw * w_sym
            sv_w = jnp.einsum('sqn,sq->sn', face_sv, wjxw_sq)
            sv_ws = jnp.einsum('sqn,sq,sqij->snij', face_sv, wjxw_sq, skew_sq)
            n_flat = face_to_surf.reshape(-1)
            w_eff = jnp.zeros(n_surf_nodes).at[n_flat].add(sv_w.reshape(-1))
            skew_eff = jnp.zeros((n_surf_nodes, 3, 3)).at[n_flat].add(
                sv_ws.reshape(-1, 3, 3)
            )
            wk = w_eff[:, None, None]

            # Physical mesh DOFs relate to reduced by u_phys = Q_node @ u_red.
            # Beam←CSR (K_bs): rows beam, cols reduced.
            # blk_t_bs ~ -k wk Q_tfm @ Q_node  (trans)
            # blk_r_bs ~ -k skew_eff @ Q_tfm? — match coil coupling_values signs
            # with csr playing the "coil" role and beam the support role.
            #
            # From coupling_values (coil←beam):
            #   blk_t_cs = -k wk Qinv
            #   blk_r_cs = +k (Qinv @ skew_eff)
            #   blk_t_sc = -k wk Q
            #   blk_r_sc = -k (skew_eff @ Q)
            # Here "coil" → CSR physical, so insert Q_node on the CSR side:
            #   K_bs (beam←csr_red): like K_sc with Q_node on the right
            #   K_sb (csr_red←beam): like K_cs with Q_node^T on the left
            blk_t_bs = (-k) * wk * (Q[None] @ Q_nodes)
            blk_r_bs = (-k) * (skew_eff @ Q[None] @ Q_nodes)
            V_parts.append(blk_t_bs.reshape(-1))
            V_parts.append(blk_r_bs.reshape(-1))

            blk_t_sb = (-k) * wk * (Q_nodes.transpose(0, 2, 1) @ Qinv[None])
            blk_r_sb = (+k) * (
                Q_nodes.transpose(0, 2, 1) @ (Qinv[None] @ skew_eff)
            )
            V_parts.append(blk_t_sb.reshape(-1))
            V_parts.append(blk_r_sb.reshape(-1))

        return jnp.concatenate(V_parts)

    # ========================================================================
    # Continuum member (CSR ring for metrics / VTU)
    # ========================================================================

    def endpoint_state(self, u_s: jax.Array) -> jax.Array:
        """Beam endpoint state; ignores the CSR DOF suffix."""
        return u_s[: 12 * self.n_beams_total].reshape(self.n_beams_total, 2, 6)

    def beam_displacement(self, geom: dict, u_s: jax.Array, xi: jax.Array):
        """Centreline displacement using only the beam DOF prefix of ``u_s``."""
        return SupportBeams.beam_displacement(
            self, geom, u_s[: 12 * self.n_beams_total], xi,
        )

    def _csr_solution(self, u_s: jax.Array, support_dofs: dict) -> list[jax.Array]:
        """Full-sector nodal CSR displacement from the reduced solve vector."""
        del support_dofs  # geometry is encoded in the reduced DOFs of u_s
        u_red = u_s[self._csr_dof_offset:].reshape(-1, 3)
        u = jnp.einsum(
            'nij,nj->ni',
            jnp.asarray(self._csr_node_Q),
            u_red[self._csr_red_node],
        )
        return [u]

    def _csr_mesh_points(self, support_dofs: dict) -> jax.Array:
        """CSR mesh node positions at the current curve DOFs."""
        return self.csr_mesh.mesh_points_from_dofs(support_dofs['csr_curve_dofs'])

    def _csr_vtu_point_data(
        self,
        curves_jax: list,
        support_dofs: dict,
        u_s: jax.Array | None,
    ) -> dict:
        """Ring-side attachment weights scattered onto full CSR mesh nodes."""
        del u_s
        geom = self.beam_geometry(curves_jax, support_dofs)
        surf_idx = np.asarray(
            self._csr_pipeline.surface_node_indices, dtype=np.int32,
        )
        csr_surf = geom['csr_points'][surf_idx]
        w_a = jnp.zeros(csr_surf.shape[0])
        for spec in self._csr_endpoint_specs(geom):
            w_k, _ = self._clamp_weights_for_spec(spec, csr_surf, support_dofs)
            w_a = w_a + w_k
        w_g = jnp.zeros_like(w_a)

        n_nodes = int(geom['csr_points'].shape[0])
        w_g_full = onp.zeros(n_nodes, dtype=onp.float64)
        w_a_full = onp.zeros(n_nodes, dtype=onp.float64)
        w_g_full[surf_idx] = onp.asarray(w_g, dtype=onp.float64)
        w_a_full[surf_idx] = onp.asarray(w_a, dtype=onp.float64)
        k_clamp = float(self.k_clamp)
        k_attach = float(self.k_attachment)
        return {
            "w_clamp": w_g_full,
            "w_attach": w_a_full,
            "k_clamp_Npm3": w_g_full * k_clamp,
            "k_attach_Npm3": w_a_full * k_attach,
        }

    @property
    def continuum_members(self) -> tuple[ContinuumMember, ...]:
        """One continuum member for the one-field-period CSR ring."""
        return (
            ContinuumMember(
                name='csr',
                pipeline=self._csr_pipeline,
                sym_weight=float(self._nfp * (1 + int(self._stellsym))),
                mesh_points=self._csr_mesh_points,
                solution=self._csr_solution,
                vtu_point_data=self._csr_vtu_point_data,
            ),
        )

    def beam_labels(self) -> tuple:
        """Per-beam coil index and type (``0``=CC, ``1``=CF, ``2``=CR)."""
        n_base = self.n_base
        parts_coil, parts_type = [], []
        for i in range(n_base):
            n_cc, n_cf, n_cr = (
                self.n_beam_cc[i], self.n_beam_cf[i], self._n_beam_cr[i],
            )
            parts_coil.append(onp.full(n_cc + n_cf + n_cr, i, dtype=onp.int32))
            parts_type.append(onp.concatenate([
                onp.zeros(n_cc, dtype=onp.int32),
                onp.ones(n_cf, dtype=onp.int32),
                onp.full(n_cr, 2, dtype=onp.int32),
            ]))
        coil_idx_arr = (
            onp.concatenate(parts_coil) if parts_coil
            else onp.zeros(0, dtype=onp.int32)
        )
        beam_type = (
            onp.concatenate(parts_type) if parts_type
            else onp.zeros(0, dtype=onp.int32)
        )
        if self.stellsym:
            n_wrap = self.n_beam_cc[n_base]
            coil_idx_arr = onp.concatenate(
                [coil_idx_arr, onp.zeros(n_wrap, dtype=onp.int32)],
            )
            beam_type = onp.concatenate(
                [beam_type, onp.zeros(n_wrap, dtype=onp.int32)],
            )
        return coil_idx_arr, beam_type
