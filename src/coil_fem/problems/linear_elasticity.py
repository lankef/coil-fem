"""3D linear isotropic elasticity for JAX-FEM.

Provides :class:`LinearElasticity3D`, a JAX-FEM ``Problem`` subclass that
supports fully-differentiable solves via ``ad_wrapper`` with respect to mesh
node positions (``params['points']``), volumetric body forces
(``params['body_force']``), and per-node Winkler spring weights
(``params['support_k']``).  Companion helpers :func:`lame_parameters`,
:func:`itc_strain`, and :func:`recompute_fe_geometry` cover common setup
tasks.
"""

from __future__ import annotations

from typing import Callable

import numpy as onp
import jax
import jax.numpy as jnp
from .device_problem import DeviceProblem
from ..metrics import (
    cauchy_stress_small_strain,
    displacement_gradient_at_quads,
    von_mises_on_quadrature,
)


# ============================================================================
# Material Properties
# ============================================================================

def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to Lamé parameters.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.

    Returns
    -------
    lam, mu : float
        First Lamé parameter [Pa] and shear modulus [Pa].
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


# ============================================================================
# Thermal eigenstrain
# ============================================================================

def itc_strain(itc: jnp.ndarray) -> jnp.ndarray:
    r"""Isotropic thermal eigenstrain from a single integral thermal contraction.

    Small-strain additive decomposition ``ε = ε_mech + ε_th`` with an isotropic
    thermal eigenstrain ``ε_th = −itc · I`` shifts the stress without changing
    the linear Hooke tangent::

        σ = λ tr(ε − ε_th) I + 2μ (ε − ε_th).

    ``itc`` (integral thermal contraction) is the (positive) dimensionless linear
    thermal contraction ``ΔL/L`` accumulated on cooldown to the service
    temperature.  Parametrising directly by the measured integral contraction
    avoids assuming a constant coefficient of thermal expansion — it is
    equivalent to ``α ΔT = −itc``.

    Parameters
    ----------
    itc : float
        Integral thermal contraction ``ΔL/L`` (e.g. ``0.0029`` for 0.29 %).

    Returns
    -------
    jnp.ndarray (3, 3)
        Thermal eigenstrain tensor ``ε_th = −itc · I``.
    """
    s = jnp.asarray(itc, dtype=float).reshape(())
    return -s * jnp.eye(3, dtype=jnp.float64)


# ============================================================================
# JAX-native FE geometry (Path C)
# ============================================================================

def recompute_fe_geometry(points, cells, shape_grads_ref, shape_vals, quad_weights):
    """Recompute FE volume geometry from mesh points using pure JAX ops.

    This is a JAX port of ``FiniteElement.get_shape_grads()`` and
    ``get_physical_quad_points()``.  All operations use ``jnp``, so JAX can
    differentiate through ``points`` when called inside ``set_params``.

    Connectivity (``cells``) and reference-element data are treated as static
    constants.

    Parameters
    ----------
    points : jnp.ndarray, (N, dim)
        Mesh node coordinates — the differentiable quantity.
    cells : array-like int, (num_cells, num_nodes)
        Element connectivity.  Integer index array, not differentiated.
    shape_grads_ref : jnp.ndarray, (num_quads, num_nodes, dim)
        Reference-element shape-function gradients.
    shape_vals : jnp.ndarray, (num_quads, num_nodes)
        Reference-element shape-function values.
    quad_weights : jnp.ndarray, (num_quads,)
        Quadrature weights.

    Returns
    -------
    shape_grads : jnp.ndarray, (num_cells, num_quads, num_nodes, dim)
        Physical shape-function gradients dN/dx.
    JxW : jnp.ndarray, (num_cells, num_quads)
        Jacobian determinant × quadrature weight.
    v_grads_JxW : jnp.ndarray, (num_cells, num_quads, num_nodes, 1, dim)
        Test-function weighted gradients (shape_grads * JxW).
    physical_quad_points : jnp.ndarray, (num_cells, num_quads, dim)
        Physical positions of quadrature points.
    """
    physical_coos = points[cells]  # (num_cells, num_nodes, dim)

    # Jacobian J = dx/d(xi): sum_nodes x_i * (dN_i/d(xi))
    # shapes: (num_cells, 1, num_nodes, dim, 1) * (1, num_quads, num_nodes, 1, dim)
    #       -> sum over node axis -> (num_cells, num_quads, 1, dim, dim)
    jacobian = jnp.sum(
        physical_coos[:, None, :, :, None] * shape_grads_ref[None, :, :, None, :],
        axis=2, keepdims=True,
    )
    det_J = jnp.linalg.det(jacobian)[:, :, 0]   # (num_cells, num_quads)
    inv_J = jnp.linalg.inv(jacobian)              # (num_cells, num_quads, 1, dim, dim)

    # dN/dx = dN/d(xi) @ J^{-1}
    # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
    # -> (num_cells, num_quads, num_nodes, 1, dim) -> squeeze last-but-one axis
    shape_grads = (shape_grads_ref[None, :, :, None, :] @ inv_J)[:, :, :, 0, :]
    # (num_cells, num_quads, num_nodes, dim)

    JxW = det_J * quad_weights[None, :]  # (num_cells, num_quads)

    v_grads_JxW = shape_grads[:, :, :, None, :] * JxW[:, :, None, None, None]
    # (num_cells, num_quads, num_nodes, 1, dim)

    physical_quad_points = jnp.sum(
        shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2
    )  # (num_cells, num_quads, dim)

    return shape_grads, JxW, v_grads_JxW, physical_quad_points


# ============================================================================
# Internal-variable ordering contract
# ============================================================================

# Positional order of entries in ``LinearElasticity3D.internal_vars``.
# JAX-FEM vmaps over these in the same order, so the kernel signatures
# ``stress(u_grad, bf, lam, mu, eps_th)`` and
# ``mass_map(u, x, bf, lam, mu, eps_th)`` must match exactly.
_INTERNAL_VAR_NAMES = ('body_force', 'lam_q', 'mu_q', 'eps_th_q')


# ============================================================================
# JAX-FEM Problem Class
# ============================================================================

class LinearElasticity3D(DeviceProblem):
    """Linear isotropic elasticity, fully differentiable via ``ad_wrapper``.

    Implements small-strain Hooke's law ``σ = λ tr(ε) I + 2μ ε``.
    ``set_params`` recomputes all FE geometry arrays from ``params['points']``
    using pure JAX, exposing gradients through mesh node positions.  Every
    exterior mesh face is automatically detected and equipped with a Winkler
    spring-foundation BC whose per-quad stiffness is supplied at run-time as
    ``params['support_k']`` (already in N/m³, owned by
    :class:`~coil_fem.coupling.Support`).  Dirichlet BCs
    (``dirichlet_bc_info``) may still be applied on top of the Winkler
    surface.

    Parameters
    ----------
    ``additional_info`` is a tuple ``(E, nu, body_force[, itc])``:

    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    body_force : tuple[float, float, float] or callable
        Body force [N/m³] for forward-only solves.  For the differentiable
        path supply ``params['body_force']`` instead.
    itc : float, optional
        Integral thermal contraction ``ΔL/L`` (positive, dimensionless).
        Pre-computes eigenstrain ``ε_th = −itc · I`` at construction.

    Examples
    --------
    Differentiable solve with Winkler BC::

        problem = LinearElasticity3D(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type,
            additional_info=(200e9, 0.3, (0., 0., 0.)),
        )
        fwd_pred = ad_wrapper(problem)
        params = {
            'points':      jnp.array(mesh.points),
            'body_force':  lorentz_at_quads,
            'support_k':   support.stiffness(w_g, w_a),
        }
        sol = fwd_pred(params)
    """

    #: Symmetry claim for the assembled stiffness matrix.
    #: Elasticity tangent, Winkler surface-mass term, and symmetric Dirichlet
    #: elimination are all symmetric unconditionally.
    matrix_symmetry: str = 'symmetric'

    #: Linearity claim: ``R(u) = K u - f`` exactly (affine in ``u``), so a
    #: single Newton step from ``u=0`` is the exact solution.  Curvature
    #: enters only through ``params['points']``, which is a parameter, not the
    #: unknown.
    is_linear: bool = True

    def custom_init(
        self,
        E: float,
        nu: float,
        body_force,
        itc: float | None = None,
    ):
        """JAX-FEM hook called by ``Problem.__init__`` after mesh setup.

        Stores material constants, pre-computes the thermal eigenstrain (if an
        ``itc`` fraction is provided), caches static reference-element
        arrays for use in :meth:`set_params`, and evaluates the initial
        body-force at quadrature points for forward-only solves.

        Parameters
        ----------
        E : float
            Young's modulus [Pa].
        nu : float
            Poisson's ratio.
        body_force : tuple[float, float, float] or callable
            Body force [N/m³].  A 3-tuple gives a uniform force; a callable
            ``f(x) -> jnp.array(3)`` gives a position-dependent force.  Used
            only for forward-only solves; for the differentiable path supply
            ``params['body_force']`` to :meth:`set_params`.
        itc : float, optional
            Integral thermal contraction ``ΔL/L`` on cooldown (positive,
            dimensionless).  Pre-computes the constant eigenstrain
            ``ε_th = −itc · I``.
        """
        self.E = float(E)
        self.nu = float(nu)
        self.lam, self.mu = lame_parameters(E, nu)

        # Thermal eigenstrain — pre-computed once since the integral thermal
        # contraction is a fixed scalar (not an optimizable DOF).  Uniform
        # contraction throughout the coil is assumed: ε_th = −itc · I.
        self.itc = float(itc) if itc is not None else None
        if self.itc is not None:
            self.epsilon_th = itc_strain(self.itc)
        else:
            self.epsilon_th = None

        # Cache static reference-element data for recompute_fe_geometry.
        fe = self.fes[0]
        self._cells_jnp = jnp.asarray(fe.cells)
        self._sg_ref = jnp.asarray(fe.shape_grads_ref)     # (num_quads, num_nodes, dim)
        self._sv     = jnp.asarray(fe.shape_vals)          # (num_quads, num_nodes)
        self._qw     = jnp.asarray(fe.quad_weights)        # (num_quads,)

        # This class always auto-detects its own exterior Winkler surface, so
        # location_fns (which would also populate boundary_inds_list) is not
        # supported.
        if len(self.boundary_inds_list) != 0:
            raise ValueError(
                "LinearElasticity3D: location_fns is not supported; the "
                "exterior Winkler surface is always auto-detected."
            )

        from jax_fem.basis import get_face_shape_vals_and_grads as _gfsv
        _ele_type = self.ele_type[0] if isinstance(self.ele_type, list) else self.ele_type
        _, _, _, _, face_corner_inds = _gfsv(_ele_type)
        cells_np = onp.asarray(fe.cells, dtype=onp.int64)
        n_ftypes = face_corner_inds.shape[0]

        # Detect exterior faces: sorted corner-node tuple appears exactly once.
        all_fc = cells_np[:, face_corner_inds].reshape(-1, face_corner_inds.shape[1])
        all_fc_sorted = onp.sort(all_fc, axis=1)
        _, inv, counts_fc = onp.unique(
            all_fc_sorted, axis=0, return_inverse=True, return_counts=True
        )
        is_ext = (counts_fc == 1)[inv]  # (n_cells * n_ftypes,)
        ext_flat = onp.where(is_ext)[0]
        correct_bi = onp.stack(
            [ext_flat // n_ftypes, ext_flat % n_ftypes], axis=1
        ).astype(onp.int32)

        # For TET10: upgrade face quadrature to gauss_order=4 BEFORE building
        # the surface arrays, so everything is built once at the right order.
        # gauss_order=2 gives only 3 face quad points for TRI6 faces, which
        # under-integrates degree-4 products of quadratic shape functions and
        # yields a rank-deficient surface stiffness.
        if _ele_type == 'TET10':
            face_sv, face_sg, face_qw, face_normals, _ = \
                _gfsv('TET10', gauss_order=4)
            fe.face_shape_vals    = face_sv
            fe.face_shape_grads_ref = face_sg
            fe.face_quad_weights  = face_qw
            fe.num_face_quads     = face_qw.shape[1]

        # Build all six boundary structures, mirroring Problem.__init__:132-157.
        s_shape_grads, n_scale, s_shape_vals = [], [], []
        for fe_i in self.fes:
            fsg_phys, ns = fe_i.get_face_shape_grads(correct_bi)
            s_shape_grads.append(fsg_phys)
            n_scale.append(ns)
            s_shape_vals.append(fe_i.face_shape_vals[correct_bi[:, 1]])
        self.boundary_inds_list         = [correct_bi]
        self.cells_list_face_list       = [
            [cells[correct_bi[:, 0]] for cells in self.cells_list]
        ]
        self.selected_face_shape_grads  = [onp.concatenate(s_shape_grads, axis=2)]
        self.nanson_scale               = [onp.transpose(onp.stack(n_scale), (1, 0, 2))]
        self.selected_face_shape_vals   = [onp.concatenate(s_shape_vals, axis=2)]
        self.physical_surface_quad_points = [
            self.fes[0].get_physical_surface_quad_points(correct_bi)
        ]
        self.internal_vars_surfaces     = [()]

        # Append face DOF contributions to the bulk-only I/J arrays.
        _d = sum(_fe.num_nodes * _fe.vec for _fe in self.fes)
        _c_face = [_cells[correct_bi[:, 0]] for _cells in self.cells_list]
        _inds_parts = []
        for _k, (_fe2, _cf) in enumerate(zip(self.fes, _c_face)):
            _crt = (
                _fe2.vec * _cf[:, :, None]
                + onp.arange(_fe2.vec)[None, None, :]
                + self.offset[_k]
            )
            _inds_parts.append(_crt.reshape(len(correct_bi), -1))
        _inds_f = onp.concatenate(_inds_parts, axis=1)
        _I_f = onp.repeat(_inds_f[:, :, None], _d, axis=2).reshape(-1)
        _J_f = onp.repeat(_inds_f[:, None, :], _d, axis=1).reshape(-1)
        self.I = onp.hstack((self.I, _I_f))
        self.J = onp.hstack((self.J, _J_f))

        # Face reference data — cached for use in set_params on every forward pass.
        self._face_sg_ref  = jnp.asarray(fe.face_shape_grads_ref)
        self._face_sv      = jnp.asarray(fe.face_shape_vals)
        self._face_qw      = jnp.asarray(fe.face_quad_weights)
        self._face_normals = jnp.asarray(fe.face_normals)

        # Pre-compute static mappings for differentiable Winkler BC.
        # These are built once here and reused in set_params every forward pass.
        self._build_winkler_surface_maps()

        # Build body-force callable for forward-only evaluation.
        if callable(body_force):
            bf_fn = body_force
        else:
            f = jnp.asarray(body_force, dtype=jnp.float64).reshape(3)
            bf_fn = lambda x: f

        # Evaluate body force at the initial quad points.  Result used as
        # internal_vars for forward-only solves; overwritten by set_params.
        pqp = jnp.asarray(self.physical_quad_points)  # (num_cells, num_quads, dim)
        bf_at_quads = jax.vmap(jax.vmap(bf_fn))(pqp)  # (num_cells, num_quads, 3)

        # Build initial per-quad material arrays for internal_vars.
        # set_params overwrites these on every differentiable forward pass.
        n_cells = fe.num_cells
        n_quads = int(self._qw.shape[0])
        lam_q_init = jnp.full((n_cells, n_quads), self.lam)
        mu_q_init  = jnp.full((n_cells, n_quads), self.mu)
        if self.epsilon_th is not None:
            eps_th_q_init = jnp.broadcast_to(
                self.epsilon_th[None, None], (n_cells, n_quads, 3, 3)
            )
        else:
            eps_th_q_init = jnp.zeros((n_cells, n_quads, 3, 3), dtype=jnp.float64)

        self.internal_vars = [bf_at_quads, lam_q_init, mu_q_init, eps_th_q_init]

    def _build_winkler_surface_maps(self):
        """Build static face-to-surface-node index maps.

        Uses ``boundary_inds_list[0]`` (the exterior surface built in
        :meth:`custom_init`).

        After this call:

        * ``self._surf_face_to_surf_node`` has shape
          ``(num_sel_faces, nodes_per_face)`` with entries in
          ``[0, n_surface_nodes)``.
        * ``self._sel_face_sv`` has shape
          ``(num_sel_faces, num_face_quads, nodes_per_face)`` — face shape
          function values, used to interpolate nodal ``u`` to surface quad
          points (:meth:`interp_surface_nodal_to_quads`) and to fold
          quad-point weights back to per-node DOF quantities (monolithic
          ``coupling_values``).
        """
        fe = self.fes[0]
        bi = self.boundary_inds_list[0]   # (num_sel, 2): [cell_idx, local_face_idx]
        cells_np = onp.asarray(fe.cells)  # (num_cells, n_cell_nodes)
        bi_np    = onp.asarray(bi)        # (num_sel, 2)

        # face_shape_vals has shape (n_face_types, n_fq, n_CELL_nodes) — it stores
        # shape-function values for ALL cell nodes at face quad points.  Interior
        # nodes have zero values on any face they do not belong to.
        # We exploit this to determine which local cell nodes are on each face type.
        face_sv_np  = onp.asarray(self._face_sv)  # (n_face_types, n_fq, n_cell_nodes)
        n_face_types = face_sv_np.shape[0]
        n_fq         = face_sv_np.shape[1]

        # For face type f, the local cell nodes with non-zero average shape values
        # are exactly the nodes on that face.
        face_type_to_local_nodes: list[onp.ndarray] = []
        for f in range(n_face_types):
            on_face = onp.abs(face_sv_np[f]).sum(axis=0) > 1e-10  # (n_cell_nodes,)
            face_type_to_local_nodes.append(onp.where(on_face)[0])

        # Sanity-check: all face types must have the same number of nodes.
        n_per_face = {len(v) for v in face_type_to_local_nodes}
        if len(n_per_face) != 1:
            raise ValueError(
                f"Inconsistent face-node counts across face types: {n_per_face}. "
                "Only uniform element types are supported for Winkler BCs."
            )
        n_face_nodes = n_per_face.pop()

        local_node_inds = onp.array(
            face_type_to_local_nodes, dtype=onp.int32
        )  # (n_face_types, n_face_nodes)

        # For each selected face, look up its n_face_nodes local → global node indices.
        local_for_each = local_node_inds[bi_np[:, 1]]   # (num_sel, n_face_nodes)
        # gfn[s, k] = cells_np[bi_np[s, 0], local_for_each[s, k]]
        # Row index broadcast: (num_sel, 1) × (num_sel, n_face_nodes) → (num_sel, n_face_nodes)
        gfn = cells_np[bi_np[:, 0:1], local_for_each]   # (num_sel, n_face_nodes)

        # Face shape vals restricted to face nodes: (num_sel, n_fq, n_face_nodes)
        # sel_all[s, q, :] has n_cell_nodes entries; we keep only the n_face_nodes
        # columns that correspond to face nodes (interior-node columns are zero anyway,
        # but restricting avoids storing unused zeros in the traced computation).
        sel_all  = face_sv_np[bi_np[:, 1]]              # (num_sel, n_fq, n_cell_nodes)
        num_sel  = len(bi_np)
        sel_face_sv_arr = sel_all[
            onp.arange(num_sel)[:, None, None],          # (num_sel, 1, 1)
            onp.arange(n_fq)[None, :, None],             # (1, n_fq, 1)
            local_for_each[:, None, :],                  # (num_sel, 1, n_face_nodes)
        ]  # (num_sel, n_fq, n_face_nodes)

        # Compact surface node indexing: map global node ids → [0, n_surf_nodes).
        unique_surf_nodes, inverse = onp.unique(gfn.ravel(), return_inverse=True)
        surf_face_to_surf_node = inverse.reshape(gfn.shape)  # (num_sel, n_face_nodes)

        self._surf_unique_global_nodes = jnp.asarray(unique_surf_nodes, dtype=jnp.int32)
        self._surf_face_to_surf_node   = jnp.asarray(surf_face_to_surf_node, dtype=jnp.int32)
        self._sel_face_sv = jnp.asarray(sel_face_sv_arr, dtype=jnp.float64)
        # (num_sel, n_fq, n_face_nodes)

    @property
    def surface_node_global_indices(self) -> jnp.ndarray:
        """Global node indices of all Winkler surface nodes.

        Shape ``(n_surface_nodes,)``.  These are the indices into the full
        ``(n_nodes, 3)`` mesh-node array used by :meth:`interp_surface_nodal_to_quads`
        and by the monolithic coupling pattern.  Not related to the shape of
        ``params['support_k']``, which is per-surface-quad.
        """
        return self._surf_unique_global_nodes

    @property
    def n_surface_quads(self) -> int:
        """Total number of surface quadrature points.

        Equal to ``num_selected_faces × num_face_quads``.  This is the leading
        dimension of ``params['support_k']`` accepted by
        :meth:`set_params`.
        """
        s = self._sel_face_sv.shape
        return int(s[0] * s[1])

    def surface_quad_points(self, points: jnp.ndarray) -> jnp.ndarray:
        """Physical positions of all surface quadrature points.

        Differentiable with respect to ``points`` (mesh node positions).

        Parameters
        ----------
        points : jnp.ndarray, shape ``(n_nodes, 3)``
            Current mesh node positions.

        Returns
        -------
        jnp.ndarray, shape ``(n_surface_quads, 3)``
            Flat array of physical quad-point coordinates, ordered face-major
            then quad-minor (consistent with the layout of
            ``params['support_k']``).
        """
        bi = self.boundary_inds_list[0]
        physical_coos = points[self._cells_jnp]           # (n_cells, n_nodes, 3)
        selected_coos = physical_coos[bi[:, 0]]            # (n_sel, n_nodes, 3)
        sel_sv        = self._face_sv[bi[:, 1]]            # (n_sel, n_fq, n_nodes)
        spqp = jnp.sum(
            sel_sv[:, :, :, None] * selected_coos[:, None, :, :], axis=2
        )  # (n_sel, n_fq, 3)
        return spqp.reshape(-1, 3)                         # (n_sq, 3)

    def surface_jxw(self, points: jnp.ndarray) -> jnp.ndarray:
        """Jacobian-times-quadrature-weight at every surface quadrature point.

        Returns the area measure ``|J| × w_q`` (Nanson scale × quad weight)
        for each face-quad pair on the Winkler surface.  Differentiable with
        respect to ``points``.

        Parameters
        ----------
        points : jnp.ndarray, shape ``(n_nodes, 3)``
            Current mesh node positions.

        Returns
        -------
        jnp.ndarray, shape ``(num_sel, n_fq)``
            JxW values, one per selected face per face quadrature point.
        """
        bi = self.boundary_inds_list[0]
        physical_coos = points[self._cells_jnp]            # (n_cells, n_nodes, 3)
        selected_coos = physical_coos[bi[:, 0]]             # (num_sel, n_nodes, 3)
        sel_grads_ref = self._face_sg_ref[bi[:, 1]]         # (num_sel, n_fq, n_nodes, 3)
        sel_normals   = self._face_normals[bi[:, 1]]        # (num_sel, 3)

        jacobian = jnp.sum(
            selected_coos[:, None, :, :, None] * sel_grads_ref[:, :, :, None, :],
            axis=2,
        )  # (num_sel, n_fq, 3, 3)
        det_J = jnp.linalg.det(jacobian)                   # (num_sel, n_fq)
        inv_J = jnp.linalg.inv(jacobian)                   # (num_sel, n_fq, 3, 3)
        ns_geom = jnp.linalg.norm(
            (sel_normals[:, None, None, :] @ inv_J)[:, :, 0, :], axis=-1
        )  # (num_sel, n_fq)
        sel_weights = self._face_qw[bi[:, 1]]
        return ns_geom * det_J * sel_weights               # (num_sel, n_fq)

    def interp_surface_nodal_to_quads(self, field: jnp.ndarray) -> jnp.ndarray:
        """Interpolate a compact surface-node field to surface quad points.

        Uses the cached face shape-function values to evaluate ``u(x_q) =
        Σ_n N_n(x_q) u_n`` for each surface quadrature point, where ``n``
        ranges over the surface-node compact index.

        Parameters
        ----------
        field : jnp.ndarray, shape ``(n_surface_nodes,)`` or ``(n_surface_nodes, d)``
            Field values at the compact surface nodes (i.e. indexed by
            :attr:`surface_node_global_indices`).

        Returns
        -------
        jnp.ndarray, shape ``(n_surface_quads,)`` or ``(n_surface_quads, d)``
            Field interpolated to every surface quadrature point.
        """
        f_face = field[self._surf_face_to_surf_node]       # (n_sel, n_face_nodes[, d])
        if field.ndim == 1:
            interp = jnp.einsum('sqn,sn->sq', self._sel_face_sv, f_face)
        else:
            interp = jnp.einsum('sqn,snd->sqd', self._sel_face_sv, f_face)
        return interp.reshape(self.n_surface_quads, *field.shape[1:])

    def get_tensor_map(self):
        """JAX-FEM hook: return the constitutive (stress) closure for volume assembly.

        JAX-FEM's ``laplace_kernel`` calls
        ``vmap(tensor_map)(u_grad, *internal_vars_per_quad)`` at every
        quadrature point, where ``internal_vars_per_quad`` are the per-point
        slices of ``self.internal_vars`` in the order defined by
        :data:`_INTERNAL_VAR_NAMES`:
        ``(body_force, lam, mu, eps_th)``.

        Returns
        -------
        stress : callable
            ``stress(u_grad, bf, lam, mu, eps_th) -> jnp.ndarray (3, 3)``

            Cauchy stress σ = λ tr(ε_m) I + 2μ ε_m, where
            ε_m = ½(∇u + ∇uᵀ) − ε_th.  Pass a zero ``(3, 3)`` ``eps_th``
            for the isothermal case.  The ``bf`` argument is ignored here
            (consumed by :meth:`get_mass_map`).
        """
        def stress(u_grad, bf, lam, mu, eps_th):
            return cauchy_stress_small_strain(u_grad, lam, mu, epsilon_th=eps_th)

        return stress

    def get_mass_map(self):
        """JAX-FEM hook: return the body-force closure for the load vector.

        JAX-FEM's ``mass_kernel`` calls
        ``vmap(mass_map)(u, x, *internal_vars)`` at every quadrature point and
        **adds** the result to the cell residual::

            residual = laplace_val + mass_val

        The elasticity weak form is ``∫ σ:∇v dV − ∫ b·v dV = 0``, so the
        contribution of the body force to the residual must be **negative**:

            mass_val  =  ∫ mass_map(u, x, bf) · v dV
                      =  ∫ (−b) · v dV

        Therefore ``mass_map`` must return ``−b`` (the negation of the
        physical body force).

        The body-force vector stored in ``internal_vars[0]`` (set via
        :meth:`custom_init` or :meth:`set_params`) is the **physical** Lorentz
        + gravity force ``b`` in [N/m³].  It is passed as the third positional
        argument ``bf`` and negated here before being returned to JAX-FEM.

        Returns
        -------
        mass_map : callable
            ``mass_map(u, x, bf, lam, mu, eps_th) -> jnp.ndarray (3,)``

            Returns ``-bf``.  JAX-FEM integrates this against the test
            function to form ``−∫ b · v dV``, matching the load term on the
            right-hand side of the weak form.  The ``lam``, ``mu``, and
            ``eps_th`` arguments mirror :meth:`get_tensor_map`'s signature
            (same ``internal_vars`` positional order) and are ignored here.
        """
        def mass_map(u, x, bf, lam, mu, eps_th):
            """bf: (3,) physical body-force [N/m³]; negate for JAX-FEM residual convention."""
            return -bf

        return mass_map

    def get_surface_maps(self):
        """Grounded Winkler spring traction: ``t = u``.

        :meth:`set_params` absorbs the per-quad stiffness (already the sum of
        the grounded-clamp and beam-attachment contributions, owned by
        :class:`~coil_fem.coupling.Support`) into ``nanson_scale`` so that the
        surface integral ``∫ k(x) u · v dS`` is computed correctly.
        """
        def spring(u, x):
            return u

        return [spring]

    def set_params(self, params):
        """Update geometry, body force, and Winkler BC from differentiable params.

        Called by ``ad_wrapper`` on every forward pass so that the JAX trace
        captures the dependency on ``params``.

        Parameters
        ----------
        params : dict
            ``'points'`` : jnp.ndarray (N, 3)
                Mesh node positions — the differentiable quantity.
            ``'body_force'`` : jnp.ndarray (num_cells, num_quads, 3)
                Body force at every quadrature point.
            ``'support_k'`` : jnp.ndarray (n_surface_quads,)
                Per-surface-quad Winkler stiffness [N/m³], applied directly
                without any interpolation.  Obtain this array via
                :meth:`surface_quad_points` → ``support.compute_weights`` →
                ``support.stiffness``.  Gradients flow through this array via
                the adjoint.
        """
        points = params['points']

        # Recompute all geometry arrays in JAX so AD can trace through points.
        # Use pre-computed geometry from the caller when available, avoiding a
        # redundant call on the CPU path (review item 2d / p5f).
        _fe_geom = params.get('_fe_geom')
        if _fe_geom is not None:
            sg, jxw, vgj, pqp = _fe_geom
        else:
            sg, jxw, vgj, pqp = recompute_fe_geometry(
                points, self._cells_jnp, self._sg_ref, self._sv, self._qw,
            )
        self.shape_grads = sg
        self.JxW = jxw[:, None, :]   # (num_cells, 1, num_quads) — num_vars=1
        self.v_grads_JxW = vgj
        self.physical_quad_points = pqp

        # Recompute surface geometry and fold Winkler stiffness into nanson_scale.
        # (Surface geometry is inlined here; recompute_fe_surface_geometry was
        # its only caller and has been removed.)  boundary_inds_list always
        # holds exactly the one auto-detected exterior surface.
        bi = self.boundary_inds_list[0]
        physical_coos = points[self._cells_jnp]           # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[bi[:, 0]]            # (num_sel, num_nodes, dim)
        sel_grads_ref = self._face_sg_ref[bi[:, 1]]        # (num_sel, num_fq, num_nodes, dim)
        sel_normals   = self._face_normals[bi[:, 1]]       # (num_sel, dim)

        jacobian = jnp.sum(
            selected_coos[:, None, :, :, None] * sel_grads_ref[:, :, :, None, :],
            axis=2,
        )  # (num_sel, num_fq, dim, dim)
        det_J = jnp.linalg.det(jacobian)                  # (num_sel, num_fq)
        inv_J = jnp.linalg.inv(jacobian)                  # (num_sel, num_fq, dim, dim)

        fsg = (
            sel_grads_ref[:, :, :, None, :] @ inv_J[:, :, None, :, :]
        )[:, :, :, 0, :]  # (num_sel, num_fq, num_nodes, dim)

        ns_geom = jnp.linalg.norm(
            (sel_normals[:, None, None, :] @ inv_J)[:, :, 0, :], axis=-1
        )  # (num_sel, num_fq)
        sel_weights = self._face_qw[bi[:, 1]]
        ns_geom = ns_geom * det_J * sel_weights

        sel_sv  = self._face_sv[bi[:, 1]]                  # (num_sel, num_fq, num_nodes)
        spqp = jnp.sum(
            sel_sv[:, :, :, None] * selected_coos[:, None, :, :], axis=2
        )  # (num_sel, num_fq, dim)

        self.selected_face_shape_grads[0]    = fsg
        self.physical_surface_quad_points[0] = spqp

        # ── Winkler stiffness ──────────────────────────────────────────
        # params['support_k'] is (n_surface_quads,) — already at quad points
        # in N/m³, no interpolation or scalar multiply needed.  Reshape to
        # (num_sel, num_fq) and absorb into nanson_scale so that the surface
        # integral ∫ k(x) u · v dS is computed correctly.
        num_sel = self._sel_face_sv.shape[0]
        num_fq  = self._sel_face_sv.shape[1]
        k_at_quad = params['support_k'].reshape(num_sel, num_fq)
        self.nanson_scale[0] = (k_at_quad * ns_geom)[:, None, :]

        # Build per-quad constitutive arrays.  Fall back to uniform scalar
        # values (broadcast) when not supplied in params — this keeps the
        # differentiable path lean for the common uniform-material case while
        # allowing spatially varying fields to be injected via params later.
        n_cells = self._cells_jnp.shape[0]
        n_quads = int(self._qw.shape[0])

        lam_q = params.get('lam_q', None)
        if lam_q is None:
            lam_q = jnp.full((n_cells, n_quads), self.lam)

        mu_q = params.get('mu_q', None)
        if mu_q is None:
            mu_q = jnp.full((n_cells, n_quads), self.mu)

        eps_th_q = params.get('eps_th_q', None)
        if eps_th_q is None:
            if self.epsilon_th is not None:
                eps_th_q = jnp.broadcast_to(
                    self.epsilon_th[None, None], (n_cells, n_quads, 3, 3)
                )
            else:
                eps_th_q = jnp.zeros((n_cells, n_quads, 3, 3), dtype=jnp.float64)

        # internal_vars order matches _INTERNAL_VAR_NAMES exactly.
        self.internal_vars = [params['body_force'], lam_q, mu_q, eps_th_q]

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def von_mises_stress(self, sol_list: list) -> jnp.ndarray:
        """Compute von Mises stress at every quadrature point.

        Uses ``self.shape_grads``, which reflects the current geometry
        including any update from :meth:`set_params`.

        Parameters
        ----------
        sol_list : list[jnp.ndarray]
            Solution from ``ad_wrapper`` / ``cudss_ad_wrapper``.

        Returns
        -------
        jnp.ndarray, (num_cells, num_quads)
            Von Mises stress [Pa].
        """
        return von_mises_on_quadrature(self, sol_list, self.lam, self.mu)

    def strain_tensors(
        self,
        sol_list: list,
        shape_grads: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute total and thermal strain tensors at every quadrature point.

        Small-strain additive decomposition ``ε = ε_elastic + ε_th`` (see
        :func:`itc_strain`).  The total strain is purely geometric,
        ``ε = ½(∇u + ∇uᵀ)``, derived from the displacement field; the thermal
        eigenstrain ``ε_th = −itc · I`` is the spatially-uniform
        constant pre-computed at construction and stored as ``self.epsilon_th``.
        The stress-producing elastic strain is recovered as
        ``ε_total − ε_thermal``.

        Parameters
        ----------
        sol_list : list[jnp.ndarray]
            Solution from ``ad_wrapper`` / ``cudss_ad_wrapper``;
            ``sol_list[0]`` has shape ``(num_nodes, 3)``.
        shape_grads : jnp.ndarray, optional
            Physical shape-function gradients ``(num_cells, num_quads,
            num_nodes, dim)``.  When ``None`` (default), ``self.shape_grads`` is
            used, which reflects the geometry of the most recent
            :meth:`set_params` call.  Pass an externally recomputed array (e.g.
            from :func:`recompute_fe_geometry`) to avoid relying on mutated
            solver state.

        Returns
        -------
        eps_total : jnp.ndarray, (num_cells, num_quads, 3, 3)
            Total strain tensor at every quadrature point.
        eps_thermal : jnp.ndarray, (3, 3)
            Uniform thermal eigenstrain.  Zeros when no thermal parameters were
            configured.  Left un-broadcast for memory efficiency; subtract it
            from ``eps_total`` (broadcasts automatically) to obtain the elastic
            strain.
        """
        u_grads = displacement_gradient_at_quads(sol_list[0], self, shape_grads=shape_grads)
        eps_total = 0.5 * (u_grads + jnp.swapaxes(u_grads, -1, -2))

        eps_th = getattr(self, 'epsilon_th', None)
        eps_thermal = (
            jnp.zeros((3, 3), dtype=eps_total.dtype) if eps_th is None else eps_th
        )
        return eps_total, eps_thermal
