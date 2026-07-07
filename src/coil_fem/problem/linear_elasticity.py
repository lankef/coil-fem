"""
3D linear isotropic elasticity for JAX-FEM.

Main API
--------
* :class:`LinearElasticity3D` - JAX-FEM ``Problem`` subclass; supports
  ``ad_wrapper`` with geometry differentiation (Path C) and differentiable
  Winkler spring-foundation BCs via ``params['support_weights']``.
* :func:`dirichlet_bc` - Build Dirichlet BC info from a selection-rule callable.

Solving
-------
Solve via ``jax_fem.solver.ad_wrapper(problem)`` (CPU, differentiable) or
:func:`coil_fem.solver.cudss.cudss_ad_wrapper` (GPU zero-copy path); both return
JAX-FEM's ``sol_list`` whose first entry ``sol_list[0]`` is the displacement
field of shape ``(num_nodes, 3)``.  :class:`~coil_fem.CoilFEM` drives this
internally.

Path C: JAX-native FE geometry
-------------------------------
:meth:`LinearElasticity3D.set_params` recomputes **all** geometry-dependent FE
arrays (``shape_grads``, ``JxW``, ``v_grads_JxW``, ``physical_quad_points``)
from ``params['points']`` using pure JAX ops.  Because ``jax_fem.solver.ad_wrapper``
differentiates the residual through ``set_params`` via ``jax.vjp``, this exposes
the gradient of the FEM solution with respect to mesh node positions without
rewriting the adjoint.

Differentiable Winkler BC
--------------------------
When ``winkler_k_scalar`` is set at construction, the Winkler spring stiffness
is provided per surface-node at run-time as ``params['support_weights']`` — a
``(n_surface_nodes,)`` JAX array of weights in ``[0, 1]``.  The actual
stiffness at each surface quadrature point is interpolated from node weights
using face shape functions and scaled by ``winkler_k_scalar``.  The weight
array is absorbed into ``nanson_scale`` so ``get_surface_maps`` returns the
trivial identity surface map; no changes to JAX-FEM assembly are needed.

For a differentiable inverse problem (e.g. optimise coil geometry + currents)::

    params = {
        'points':          jnp.array(mesh.points),   # (N, 3)  from CoilMesh
        'body_force':      lorentz_at_quads,          # (num_cells, num_quads, 3)
        'support_weights': weights,                   # (n_surface_nodes,) optional
    }
    fwd_pred = ad_wrapper(problem)
    sol = fwd_pred(params)   # differentiable w.r.t. all params entries

Post-processing
---------------
* :meth:`LinearElasticity3D.von_mises_stress` - Per-quadrature-point von Mises
  stress.
* :meth:`LinearElasticity3D.strain_tensors` - Total and thermal strain tensors.
"""

from __future__ import annotations

from typing import Callable

import numpy as onp
import jax
import jax.numpy as jnp
from .device_problem import DeviceProblem


# ============================================================================
# Material Properties
# ============================================================================

def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to Lamé parameters."""
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
    jnp.ndarray, shape (3, 3)
    """
    s = jnp.asarray(itc, dtype=float).reshape(())
    return -s * jnp.eye(3, dtype=jnp.float64)


# ============================================================================
# Boundary Condition Helpers
# ============================================================================

def dirichlet_bc(
    mesh,
    selection_rule: Callable[[onp.ndarray], onp.ndarray],
    value: float = 0.0,
) -> list:
    """Build Dirichlet BC info from a node-selection callable.

    The selection rule is evaluated **once** at construction time against the
    current mesh points and the resulting boolean mask is closed over, so the
    returned location functions are cheap to evaluate during JAX-FEM init.

    Parameters
    ----------
    mesh : jax_fem.generate_mesh.Mesh
        Mesh whose ``points`` array is used to evaluate the rule.
    selection_rule : callable
        Maps ``(N, 3)`` node coordinates to a ``(N,)`` bool array.
    value : float
        Fixed displacement value applied to all three components (default 0).

    Returns
    -------
    list
        ``[location_fns, vecs, value_fns]`` in JAX-FEM Dirichlet format.
        All three displacement components (x, y, z) are fixed to ``value``.

    Example
    -------
    ::

        rule = lambda pts: pts[:, 2] < pts[:, 2].min() + 1e-5
        dbc = dirichlet_bc(mesh, rule)
        problem = LinearElasticity3D(..., dirichlet_bc_info=dbc)
    """
    flags = jnp.asarray(selection_rule(onp.asarray(mesh.points)), dtype=bool)

    def loc(p, ind):
        return flags[ind]

    val_fn = lambda p: float(value)
    return [[loc, loc, loc], [0, 1, 2], [val_fn, val_fn, val_fn]]


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
# JAX-FEM Problem Class
# ============================================================================

class LinearElasticity3D(DeviceProblem):
    """Linear isotropic elasticity with JAX-native FE geometry (Path C).

    Implements small-strain Hooke's law: σ = λ tr(ε) I + 2μ ε.

    ``set_params`` recomputes all geometry-dependent FE arrays from
    ``params['points']`` using pure JAX, so ``jax_fem.solver.ad_wrapper``
    propagates gradients through mesh node positions without any changes to the
    adjoint machinery.

    Parameters passed via ``additional_info``
    -----------------------------------------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    body_force : tuple(float, float, float) or callable(x) -> (3,)
        Body force [N/m³].  For a constant force (e.g. gravity) pass a 3-tuple.
        For a position-dependent force pass a callable ``f(x) -> jnp.array(3)``.
        This is used for the **forward-only** path; for the differentiable path
        pass ``params['body_force']`` to ``set_params`` instead.
    winkler_k_scalar : float, optional
        Base Winkler spring stiffness [N/m³].  When set, ``custom_init``
        detects exterior faces topologically and builds the Winkler surface
        from scratch — no ``location_fns`` needed.  Per-node weights in
        ``[0, 1]`` are supplied at run-time through
        ``params['support_weights']`` in :meth:`set_params`; the actual
        per-quad stiffness is ``winkler_k_scalar * interp(support_weights)``.
    itc : float, optional
        Isotropic **integral thermal contraction**: the (positive) dimensionless
        linear thermal contraction ``ΔL/L`` accumulated on cooldown to the
        service temperature.  When given, the thermal eigenstrain
        ``ε_th = −itc · I`` is pre-computed once at construction and baked
        into the constitutive law.  ``itc`` is not a differentiable DOF.
        This replaces the former ``(alpha, T_init, T_final)`` triple and is
        parametrised directly by the measured integral contraction (no
        constant-CTE assumption).

    Example — forward-only (no Winkler BC)
    ----------------------------------------
    Pass the **physical** body force (positive downward = negative z for
    gravity).  :meth:`get_mass_map` internally negates it to match JAX-FEM's
    residual convention ``R = laplace_val + mass_val``::

        from jax_fem.solver import solver

        rule = lambda pts: pts[:, 2] < pts[:, 2].min() + 1e-5
        problem = LinearElasticity3D(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type,
            dirichlet_bc_info=dirichlet_bc(mesh, rule),
            additional_info=(200e9, 0.3, (0, 0, -7800 * 9.81)),
        )
        sol = solver(problem, solver_options={"umfpack_solver": {}})

    Example — differentiable path with Winkler BC
    -----------------------------------------------
    Pass ``winkler_k_scalar`` in ``additional_info``; no ``location_fns``
    required — exterior faces are detected automatically::

        problem = LinearElasticity3D(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type,
            additional_info=(200e9, 0.3, (0., 0., 0.), 1e9),  # winkler_k_scalar=1e9
        )
        fwd_pred = ad_wrapper(problem)
        params = {
            'points':          jnp.array(mesh.points),
            'body_force':      lorentz_at_quads,
            'support_weights': weights,   # (n_surface_nodes,) in [0, 1]
        }
        sol = fwd_pred(params)   # differentiable w.r.t. all entries
    """

    def custom_init(
        self,
        E: float,
        nu: float,
        body_force,
        winkler_k_scalar: float | None = None,
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
        winkler_k_scalar : float, optional
            Base Winkler spring stiffness [N/m³].  When set, exterior surface
            faces are detected topologically and all six boundary structures
            are built here — no ``location_fns`` needed.  Per-node weights in
            ``[0, 1]`` are given at run-time via ``params['support_weights']``
            in :meth:`set_params`.
        itc : float, optional
            Integral thermal contraction ``ΔL/L`` on cooldown (positive,
            dimensionless).  Pre-computes the constant eigenstrain
            ``ε_th = −itc · I``.
        """
        self.E = float(E)
        self.nu = float(nu)
        self.lam, self.mu = lame_parameters(E, nu)
        self._winkler_k_scalar = float(winkler_k_scalar) if winkler_k_scalar is not None else None

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

        # ── Boundary Condition Construction ──────────────────────────────────
        #
        # All BC surface structures are built here, in one place, from scratch.
        # Problem.__init__ was called with location_fns=None, so boundary_inds_list
        # is empty and I/J contain only the bulk volume block.
        #
        # To add a second BC surface type in the future (e.g. a Dirichlet lift
        # surface or a Neumann traction), append another block here following
        # the same pattern: detect faces → (optionally upgrade quadrature) →
        # build the six boundary structures → append face I/J.
        #
        # ── Winkler exterior-surface BC ───────────────────────────────────────
        if self._winkler_k_scalar is not None:
            if len(self.boundary_inds_list) != 0:
                raise ValueError(
                    "LinearElasticity3D: winkler_k_scalar is set but "
                    "boundary_inds_list is non-empty (location_fns was also "
                    "passed). Use either location_fns or winkler_k_scalar, "
                    "not both."
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
                from jax_fem.basis import get_face_shape_vals_and_grads
                face_sv, face_sg, face_qw, face_normals, _ = \
                    get_face_shape_vals_and_grads('TET10', gauss_order=4)
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
        # Only needed when a Winkler surface was registered above.
        if self._winkler_k_scalar is not None:
            self._face_sg_ref  = jnp.asarray(fe.face_shape_grads_ref)
            self._face_sv      = jnp.asarray(fe.face_shape_vals)
            self._face_qw      = jnp.asarray(fe.face_quad_weights)
            self._face_normals = jnp.asarray(fe.face_normals)
        else:
            self._face_sg_ref  = None
            self._face_sv      = None
            self._face_qw      = None
            self._face_normals = None

        # Pre-compute static mappings for differentiable Winkler BC.
        # These are built once here and reused in set_params every forward pass.
        if self._winkler_k_scalar is not None:
            self._build_winkler_surface_maps()
        else:
            self._surf_face_to_surf_node = None
            self._sel_face_sv = None

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
        self.internal_vars = [bf_at_quads]

    def _build_winkler_surface_maps(self):
        """Build static face-to-surface-node index map for differentiable k.

        Uses ``boundary_inds_list[0]`` (the exterior surface built in
        :meth:`custom_init`).

        After this call ``self._surf_face_to_surf_node`` has shape
        ``(num_sel_faces, nodes_per_face)`` with entries in
        ``[0, n_surface_nodes)``, and ``self._sel_face_sv`` has shape
        ``(num_sel_faces, num_face_quads, nodes_per_face)`` — the face shape
        function values that map node weights to quad-point weights.
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
        # (num_sel, n_fq, n_face_nodes) — used in set_params to interpolate node weights
        # to quadrature-point stiffness values.

    @property
    def surface_node_global_indices(self) -> jnp.ndarray | None:
        """Global node indices of all Winkler surface nodes, or ``None``.

        Shape ``(n_surface_nodes,)``.  These are the indices into the full
        ``(n_nodes, 3)`` mesh node array that correspond to the entries of the
        ``support_weights`` vector passed to :meth:`set_params`.
        """
        return self._surf_unique_global_nodes if self._surf_face_to_surf_node is not None else None

    def get_tensor_map(self):
        """JAX-FEM hook: return the constitutive (stress) closure for volume assembly.

        JAX-FEM's ``laplace_kernel`` calls ``vmap(tensor_map)(u_grad, *internal_vars)``
        at every quadrature point to build the stiffness residual.  The returned
        closure must have the signature ``stress(u_grad, *args) -> (vec, vec)``
        where ``u_grad`` is the displacement gradient ``(3, 3)`` at a single point.

        Returns
        -------
        stress : callable
            ``stress(u_grad, *args) -> jnp.ndarray (3, 3)``

            Cauchy stress tensor at a quadrature point:

            * **Isothermal:**  σ = λ tr(ε) I + 2μ ε,  where ε = ½(∇u + ∇uᵀ).
            * **Thermoelastic:** σ = λ tr(ε − ε_th) I + 2μ (ε − ε_th),
              with the pre-computed constant eigenstrain ε_th.

            The ``*args`` catch-all absorbs body-force or other internal
            variables passed by the assembler; the stress closure ignores them.
        """
        lam, mu = self.lam, self.mu
        epsilon_th = self.epsilon_th  # None or constant (3, 3) array

        # JAX-FEM's laplace_kernel calls vmap(tensor_map)(u_grad, *cell_internal_vars).
        # Body force lives in internal_vars for get_mass_map only; σ(u) ignores it.
        if epsilon_th is None:
            def stress(u_grad, *args):
                eps = 0.5 * (u_grad + u_grad.T)
                tr = jnp.trace(eps)
                return lam * tr * jnp.eye(3, dtype=u_grad.dtype) + 2.0 * mu * eps
        else:
            def stress(u_grad, *args):
                eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
                eps_m = eps - epsilon_th
                tr = jnp.trace(eps_m, axis1=-2, axis2=-1)
                eye = jnp.eye(3, dtype=u_grad.dtype)
                return lam * tr[..., None, None] * eye + 2.0 * mu * eps_m

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
            ``mass_map(u, x, bf) -> jnp.ndarray (3,)``

            Returns ``-bf``.  JAX-FEM integrates this against the test
            function to form ``−∫ b · v dV``, matching the load term on the
            right-hand side of the weak form.
        """
        def mass_map(u, x, bf):
            """bf: (3,) physical body-force [N/m³]; negate for JAX-FEM residual convention."""
            return -bf

        return mass_map

    def get_surface_maps(self):
        """Winkler spring: identity traction ``t = u``; stiffness lives in nanson_scale.

        When ``winkler_k_scalar`` is set, :meth:`set_params` absorbs the
        per-quad stiffness into ``nanson_scale`` so that the surface integral
        ``∫ k(x) u · v dS`` is computed correctly without modifying JAX-FEM's
        assembly machinery.  The surface map therefore returns ``u`` unchanged.

        If no Winkler BC is configured (``winkler_k_scalar=None``), returns an
        empty list.
        """
        if self._winkler_k_scalar is None:
            return []

        def identity_spring(u, x):
            return u

        return [identity_spring]

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
            ``'support_weights'`` : jnp.ndarray (n_surface_nodes,), optional
                Per-surface-node Winkler weights in ``[0, 1]``.  Required when
                ``winkler_k_scalar`` was set at construction.  The stiffness at
                each surface quad point is
                ``winkler_k_scalar * interp(support_weights)``
                where the interpolation uses face shape functions.
                Gradients flow through this array via the adjoint.
        """
        points = params['points']

        # Recompute all geometry arrays in JAX so AD can trace through points.
        sg, jxw, vgj, pqp = recompute_fe_geometry(
            points, self._cells_jnp, self._sg_ref, self._sv, self._qw,
        )
        self.shape_grads = sg
        self.JxW = jxw[:, None, :]   # (num_cells, 1, num_quads) — num_vars=1
        self.v_grads_JxW = vgj
        self.physical_quad_points = pqp

        # Recompute surface geometry and fold Winkler stiffness into nanson_scale.
        # (Surface geometry is inlined here; recompute_fe_surface_geometry was
        # its only caller and has been removed.)
        for i, bi in enumerate(self.boundary_inds_list):
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

            self.selected_face_shape_grads[i]    = fsg
            self.physical_surface_quad_points[i] = spqp

            if (i == 0
                    and self._winkler_k_scalar is not None
                    and 'support_weights' in params
                    and self._surf_face_to_surf_node is not None):
                # Interpolate per-node weights to per-quad stiffness using face
                # shape functions, then absorb into nanson_scale so that the
                # identity surface map (t = u) yields ∫ k(x) u·v dS correctly.
                #
                # sel_face_sv : (num_sel, num_fq, npf)
                # w_face      : (num_sel, npf)   — weights at face corner nodes
                # k_at_quad   : (num_sel, num_fq) — scalar stiffness per quad pt
                w = params['support_weights']          # (n_surface_nodes,) traced
                w_face = w[self._surf_face_to_surf_node]   # (num_sel, npf) traced
                k_at_quad = self._winkler_k_scalar * jnp.einsum(
                    'sqn,sn->sq', self._sel_face_sv, w_face
                )  # (num_sel, num_fq) traced
                self.nanson_scale[i] = (k_at_quad * ns_geom)[:, None, :]
            else:
                self.nanson_scale[i] = ns_geom[:, None, :]

        # Body force enters the residual via internal_vars (standard path).
        self.internal_vars = [params['body_force']]

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
        fe = self.fes[0]
        cells_sol = sol_list[0][fe.cells]  # (num_cells, num_nodes, 3)
        shape_grads = self.shape_grads  # (num_cells, num_quads, num_nodes, dim)

        # u_grads[c, q, i, j] = du_i/dx_j
        u_grads = jnp.sum(
            cells_sol[:, None, :, :, None] * shape_grads[:, :, :, None, :], axis=2
        )  # (num_cells, num_quads, 3, 3)

        lam, mu = self.lam, self.mu
        epsilon_th = getattr(self, 'epsilon_th', None)  # None or constant (3, 3)

        def vm_at_point(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            eps_m = eps - epsilon_th if epsilon_th is not None else eps
            tr = jnp.trace(eps_m)
            sigma = lam * tr * jnp.eye(3, dtype=u_grad.dtype) + 2.0 * mu * eps_m
            s = sigma - (jnp.trace(sigma) / 3.0) * jnp.eye(3, dtype=u_grad.dtype)
            return jnp.sqrt(1.5 * jnp.sum(s * s) + 1e-30)

        return jax.vmap(jax.vmap(vm_at_point))(u_grads)  # (num_cells, num_quads)

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
        fe = self.fes[0]
        cells_sol = sol_list[0][fe.cells]  # (num_cells, num_nodes, 3)
        if shape_grads is None:
            shape_grads = self.shape_grads  # (num_cells, num_quads, num_nodes, dim)

        # u_grads[c, q, i, j] = du_i/dx_j
        u_grads = jnp.sum(
            cells_sol[:, None, :, :, None] * shape_grads[:, :, :, None, :], axis=2
        )  # (num_cells, num_quads, 3, 3)

        eps_total = 0.5 * (u_grads + jnp.swapaxes(u_grads, -1, -2))

        eps_th = getattr(self, 'epsilon_th', None)
        eps_thermal = (
            jnp.zeros((3, 3), dtype=eps_total.dtype) if eps_th is None else eps_th
        )
        return eps_total, eps_thermal
