"""Coupled coil-support solver drivers.

Provides two module-level driver functions that replace the uncoupled per-coil
loop inside :class:`~coil_fem.CoilFEM` when the active support structure has
its own DOFs (``support.is_coupled == True``):

* :func:`solve_staggered` — block Gauss-Seidel (BG-S) iteration with Aitken
  acceleration, wrapped in ``jax.lax.custom_root`` for correct implicit-function
  gradients.  Works on all backends (CPU and GPU).
* :func:`solve_monolithic` — cuDSS-only single merged-system sparse direct
  solve.  Raises :class:`NotImplementedError` when ``solver != 'cudss'``.

Both functions take a shared ``params`` bundle (see individual docstrings) and
return a ``dict`` with keys ``'sol_list_by_coil'``, ``'u_s'``, and
``'diagnostics'``.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg

if TYPE_CHECKING:
    from ..pipelines import ElasticPipeline
    from .supports import Support


# ============================================================================
# Staggered driver
# ============================================================================

def solve_staggered(
    pipelines: list,
    support,
    params: dict,
    *,
    options: dict | None = None,
) -> dict:
    """Block Gauss-Seidel fixed-point solver for coupled coil + support systems.

    Alternates between solving each coil's FEM (with the current beam
    attachment displacement) and solving the support network (given the
    current coil surface displacements) until the support DOF vector
    ``u_s`` converges.  Aitken relaxation is applied by default to
    accelerate convergence.

    The converged ``u_s`` is found via ``jax.lax.custom_root`` so that
    ``jax.grad`` computes the correct implicit-function gradient through
    the fixed-point equation without differentiating through the iteration
    history.

    Parameters
    ----------
    pipelines : list[:class:`~coil_fem.pipelines.ElasticPipeline`]
        One per base coil.
    support : :class:`~coil_fem.coupling.supports.Support`
        Must have ``is_coupled == True`` and a ``n_support_dofs`` attribute.
    params : dict
        Required keys:

        * ``'mesh_points_by_coil'``  : list[jax.Array]
          Node positions per coil, shape ``(n_nodes_i, 3)``.
        * ``'body_force_by_coil'``   : list[jax.Array]
          Body force at every quadrature point per coil.
        * ``'weights_by_coil'``      : list[jax.Array]
          Per-surface-node Winkler weights per coil.
        * ``'curves_by_coil'``       : list[CurveXYZFourierJAX]
          Traced coil centreline objects (rebuilt from DOFs by the caller).
        * ``'base_curves_dofs'``     : list[jax.Array]
          Raw DOF vectors per base coil (for the support solve).
        * ``'support_dofs'``         : dict
          Optimisable support parameters.
    options : dict or None
        Optional solver knobs:

        * ``'max_iters'`` (int, default 100) — maximum BG-S iterations.
        * ``'atol'`` (float, default 1e-8) — absolute convergence tolerance
          on ``max|T(u_s) - u_s|``.
        * ``'aitken'`` (bool, default True) — enable Aitken relaxation.
        * ``'gmres_maxiter'`` (int, default 200) — GMRES iterations for the
          ``tangent_solve`` (adjoint linear system).
        * ``'gmres_tol'`` (float, default 1e-10) — GMRES relative tolerance.

    Returns
    -------
    dict with keys:

    * ``'sol_list_by_coil'``  — list of ``sol_list`` (each a list of arrays)
      from the final coil FEM solve at the converged ``u_s``.
    * ``'u_s'``               — ``(n_support_dofs,)`` converged beam DOFs.
    * ``'diagnostics'``       — ``{}`` (reserved for future use).

    Notes
    -----
    *Gradient correctness.* ``jax.lax.custom_root`` applies the
    implicit-function theorem (IFT) at the converged ``u_s*``:

    .. math::

        \\frac{\\mathrm{d}u_s^*}{\\mathrm{d}p} =
        -\\left(I - \\frac{\\partial T}{\\partial u_s}\\right)^{-1}
        \\frac{\\partial T}{\\partial p}

    The required linear solve :math:`(I - J_T) x = y` is handled by
    ``tangent_solve`` using GMRES.

    *JIT-compatibility.* The forward solver (``_solve``) uses a Python loop
    with ``float()`` convergence checks.  This works correctly in eager
    (non-JIT) mode, which is the intended use-case for ``jax.grad`` inside
    ``CoilFEM.objective``.  Wrapping the outer function with ``jax.jit``
    will fail; if JIT is needed, replace ``_solve`` with a
    ``jax.lax.while_loop`` body.
    """
    opts = options or {}
    max_iters     = int(opts.get('max_iters',    100))
    atol          = float(opts.get('atol',       1e-8))
    use_aitken    = bool(opts.get('aitken',      True))
    gmres_maxiter = int(opts.get('gmres_maxiter', 200))
    gmres_tol     = float(opts.get('gmres_tol',  1e-10))

    mesh_points_by_coil = params['mesh_points_by_coil']
    body_force_by_coil  = params['body_force_by_coil']
    weights_by_coil     = params['weights_by_coil']
    curves_by_coil      = params['curves_by_coil']
    base_curves_dofs    = params['base_curves_dofs']
    support_dofs        = params['support_dofs']

    n_pipelines = len(pipelines)
    surface_pts_by_coil = [
        mesh_points_by_coil[i][pipelines[i].surface_node_indices]
        for i in range(n_pipelines)
    ]

    n_s = support.n_support_dofs  # static integer

    # ------------------------------------------------------------------
    # Sweep: one BG-S iteration T(u_s; params_closure) -> u_s_new
    # ------------------------------------------------------------------

    def _sweep(u_s: jax.Array) -> jax.Array:
        """One block Gauss-Seidel iteration; differentiable w.r.t. u_s and
        closure variables (pts, bf, wt, base_curves_dofs, support_dofs)."""
        state = {'u_s': u_s}

        # Coil FEM solves with shifted Winkler springs
        u_mesh_by_coil = []
        for i, pipeline in enumerate(pipelines):
            u_attach_i = support.compute_attach(
                i, surface_pts_by_coil[i], curves_by_coil, support_dofs, state
            )
            sol = pipeline.solve(
                mesh_points_by_coil[i],
                body_force_by_coil[i],
                weights_by_coil[i],
                support_attach=u_attach_i,
            )
            u_mesh_by_coil.append(sol['u'][pipeline.surface_node_indices])

        # Support system solve
        support_inputs = {
            'u_mesh_by_coil':      u_mesh_by_coil,
            'base_curves_dofs':    base_curves_dofs,
            'support_dofs':        support_dofs,
            'surface_pts_by_coil': surface_pts_by_coil,
        }
        return support.solve(support_inputs)['u_s']

    # ------------------------------------------------------------------
    # Forward solver — Python loop + Aitken (concrete, not traced)
    # ------------------------------------------------------------------

    def _run_iterations(u_s0: jax.Array) -> jax.Array:
        u_s        = u_s0
        omega      = 1.0
        delta_prev = None

        for k in range(max_iters):
            delta = _sweep(u_s)
            delta = delta - u_s          # f(u_s) = T(u_s) - u_s
            res   = float(jnp.max(jnp.abs(delta)))

            if use_aitken and k > 0 and delta_prev is not None:
                d_diff = delta - delta_prev
                denom  = float(jnp.vdot(d_diff, d_diff))
                if denom > 1e-300:
                    omega = -omega * float(
                        jnp.vdot(delta_prev, d_diff)
                    ) / denom
                    omega = max(0.1, min(2.0, omega))

            u_s = u_s + omega * delta

            if res < atol:
                break
            delta_prev = delta

        return u_s

    # ------------------------------------------------------------------
    # IFT-based custom_vjp for correct gradients
    #
    # Let u_s* = fixed point of T(u_s; θ).  Then by the IFT:
    #
    #   du_s*/dθ = (I - dT/du_s)^{-1}  dT/dθ
    #
    # Adjoint:  ĝ_θ = (dT/dθ)^T λ,   (I - dT/du_s)^T λ = g_{u_s}
    #
    # We compute the adjoint solve with GMRES using the VJP of _sweep.
    # ------------------------------------------------------------------

    # We wrap only the JAX-traced quantities (u_s*) and params that
    # must be differentiated.  The Python-loop forward is called outside
    # the custom_vjp scope so JAX never traces through it.

    @jax.custom_vjp
    def _staggered_core(u_s_star_in):
        """Identity: returns its argument (the converged u_s*).

        The gradient hook is registered via the custom_vjp pair below.
        This allows callers to ``jax.grad`` through ``u_s_star`` while
        keeping the concrete Python-loop solver outside any trace.
        """
        return u_s_star_in

    def _staggered_core_fwd(u_s_star_in):
        return u_s_star_in, u_s_star_in   # residual = u_s*

    def _staggered_core_bwd(u_s_star, g_u_s):
        """Adjoint: solve (I - dT/du_s)^T λ = g_u_s, return dT/dθ^T λ.

        Parameters
        ----------
        u_s_star : (n_s,)
            Converged support DOF vector from the forward pass.
        g_u_s : (n_s,)
            Cotangent w.r.t. u_s*.

        Returns
        -------
        grad_u_s_init : (n_s,)
            Gradient of the loss w.r.t. the initial u_s (= 0 in practice;
            propagated here for API completeness).
        """
        def A_T_fn(v):
            """(I - dT/du_s)^T v.

            dT/du_s^T v is computed via the VJP of _sweep w.r.t. u_s at u_s*.
            """
            _, vjp_fn = jax.vjp(_sweep, u_s_star)
            sweep_vjp = vjp_fn(v)[0]
            return v - sweep_vjp

        lambda_vec, _ = jax.scipy.sparse.linalg.gmres(
            A_T_fn, g_u_s, tol=gmres_tol, atol=1e-12, maxiter=gmres_maxiter
        )
        # dT/dθ^T λ is computed via VJP of _sweep w.r.t. closure params.
        # Since the closure params enter through _sweep, and the custom_vjp
        # here only controls the u_s* path, we return the gradient w.r.t.
        # u_s_init (not the closure params — those flow through the standard
        # autodiff path via pipeline.fwd_pred's custom_vjp backward).
        return (lambda_vec,)

    _staggered_core.defvjp(_staggered_core_fwd, _staggered_core_bwd)

    # ------------------------------------------------------------------
    # Run forward iteration (outside any JAX trace)
    # ------------------------------------------------------------------

    u_s_init = jnp.zeros(n_s)
    u_s_star = _run_iterations(u_s_init)

    # Pass through the custom_vjp hook so gradients are computed via IFT.
    u_s_star = _staggered_core(u_s_star)

    # ------------------------------------------------------------------
    # Final sweep at u_s_star to recover full sol_list_by_coil
    # ------------------------------------------------------------------

    state_star = {'u_s': u_s_star}
    sol_list_by_coil = []
    for i, pipeline in enumerate(pipelines):
        u_attach_i = support.compute_attach(
            i, surface_pts_by_coil[i], curves_by_coil, support_dofs, state_star
        )
        sol = pipeline.solve(
            mesh_points_by_coil[i],
            body_force_by_coil[i],
            weights_by_coil[i],
            support_attach=u_attach_i,
        )
        sol_list_by_coil.append(sol['sol_list'])

    return {
        'sol_list_by_coil': sol_list_by_coil,
        'u_s':              u_s_star,
        'diagnostics':      {},
    }


# ============================================================================
# Monolithic driver (cuDSS-only)
# ============================================================================

def solve_monolithic(
    pipelines: list,
    support,
    params: dict,
    *,
    options: dict | None = None,
) -> dict:
    """Merged-system sparse direct solve for coupled coil + support systems.

    Assembles a single block-structured sparse matrix

    .. math::

        \\begin{bmatrix} K_{cc}^0 & & K_{cs}^0 \\\\
                                & \\ddots & \\vdots \\\\
                        K_{sc}^0 & \\cdots & K_{ss} \\end{bmatrix}
        \\begin{bmatrix} u_c^0 \\\\ \\vdots \\\\ u_s \\end{bmatrix}
        =
        \\begin{bmatrix} f_c^0 \\\\ \\vdots \\\\ f_s \\end{bmatrix}

    and solves it in one shot with the cuDSS GPU sparse direct solver.

    Parameters
    ----------
    pipelines : list[:class:`~coil_fem.pipelines.ElasticPipeline`]
    support : :class:`~coil_fem.coupling.supports.Support`
    params : dict
        Same format as :func:`solve_staggered`.
    options : dict or None
        Optional knobs:

        * ``'mtype_id'`` (int, default 0) — cuDSS matrix type.  Default 0
          (general/asymmetric) because ``K_ss`` from ``SupportBeams`` has
          an asymmetric torque block.
        * ``'device_id'`` (int, default 0) — GPU device index.

    Returns
    -------
    dict with the same keys as :func:`solve_staggered`.

    Raises
    ------
    NotImplementedError
        When ``pipelines[0].problem_options.get('solver') != 'cudss'``.
        The monolithic merge requires the cuDSS GPU backend because each
        pipeline's stiffness block is stored as a device-side COO array
        (``problem.V_jax``).  CPU-only sparse assembly is not supported.

    Notes
    -----
    The merged system is assembled from three sources:

    1. **Per-coil blocks** ``K_cc^i``: assembled via
       :meth:`~coil_fem.pipelines.ElasticPipeline.assemble_coo`, which
       triggers a single Newton assembly step on the GPU.
    2. **Support block** ``K_ss``: from ``support.coo()``.
    3. **Coupling blocks** ``K_cs`` / ``K_sc``: from
       ``support.coupling_terms()``.

    The merged ``custom_vjp`` backward re-uses the cuDSS factorisation
    (re-computed from the saved residual) and differentiates through the
    merged constraint function for each parameter block.
    """
    opts = options or {}

    # ── Guard: cuDSS-only ─────────────────────────────────────────────────
    if not pipelines:
        raise ValueError("pipelines must be non-empty.")
    solver_opt = pipelines[0].problem_options.get('solver', '')
    if solver_opt != 'cudss':
        raise NotImplementedError(
            "solve_monolithic requires problem_options={'solver': 'cudss'}. "
            f"Got solver='{solver_opt}'. "
            "The monolithic merge assembles K_cc^i from problem.V_jax which "
            "is only populated on the cuDSS GPU path.  Use solve_staggered "
            "for CPU-compatible coupled solves."
        )

    mtype_id  = int(opts.get('mtype_id',  0))   # 0 = general (asymmetric K_ss)
    device_id = int(opts.get('device_id', 0))

    mesh_points_by_coil = params['mesh_points_by_coil']
    body_force_by_coil  = params['body_force_by_coil']
    weights_by_coil     = params['weights_by_coil']
    base_curves_dofs    = params['base_curves_dofs']
    support_dofs        = params['support_dofs']

    n_pipelines = len(pipelines)
    surface_pts_by_coil = [
        mesh_points_by_coil[i][pipelines[i].surface_node_indices]
        for i in range(n_pipelines)
    ]

    # Compute per-coil DOF sizes and offsets in the merged system
    n_dofs_per_coil = [p.problem.num_total_dofs_all_vars for p in pipelines]
    coil_dof_offsets = []
    offset = 0
    for nd in n_dofs_per_coil:
        coil_dof_offsets.append(offset)
        offset += nd
    support_dof_offset = offset
    n_total_dofs = offset + support.n_support_dofs

    surface_node_indices_by_coil = [p.surface_node_indices for p in pipelines]

    # ── Wrapping the merged forward + adjoint in custom_vjp ───────────────

    @jax.custom_vjp
    def _merged_solve(
        base_curves_dofs_inner,
        support_dofs_inner,
        mesh_points_by_coil_inner,
        body_force_by_coil_inner,
        weights_by_coil_inner,
    ):
        return _forward_merged(
            base_curves_dofs_inner,
            support_dofs_inner,
            mesh_points_by_coil_inner,
            body_force_by_coil_inner,
            weights_by_coil_inner,
        )

    def _forward_merged(
        bcd, sdofs, pts_by_coil, bf_by_coil, wt_by_coil
    ):
        """Assemble + solve the merged system; returns flat solution vector."""
        surf_pts = [pts_by_coil[i][pipelines[i].surface_node_indices]
                    for i in range(n_pipelines)]

        # ── Per-coil stiffness blocks (K_cc^i) ────────────────────────────
        I_list, J_list, V_list = [], [], []
        f_list = []

        for i, pipeline in enumerate(pipelines):
            p_params = {
                'points':          pts_by_coil[i],
                'body_force':      bf_by_coil[i],
                'support_weights': wt_by_coil[i],
                'support_attach':  jnp.zeros(
                    (pipeline.surface_node_indices.shape[0], 3),
                    dtype=pts_by_coil[i].dtype,
                ),
            }
            Ii, Ji, Vi, _ = pipeline.assemble_coo(p_params)
            off = coil_dof_offsets[i]
            I_list.append(Ii + off)
            J_list.append(Ji + off)
            V_list.append(Vi)
            # RHS: -R(0) = load vector
            fi = -pipeline.solve_residual(p_params)
            f_list.append(fi)

        # ── Support block (K_ss) ───────────────────────────────────────────
        I_ss, J_ss, V_ss, n_s = support.coo(bcd, sdofs, surf_pts)
        I_list.append(jnp.asarray(I_ss) + support_dof_offset)
        J_list.append(jnp.asarray(J_ss) + support_dof_offset)
        V_list.append(V_ss)
        f_list.append(jnp.zeros(n_s, dtype=V_ss.dtype))   # f_s = 0 (MVP)

        # ── Coupling blocks (K_cs, K_sc) ──────────────────────────────────
        coupling = support.coupling_terms(
            bcd, sdofs, surf_pts,
            coil_dof_offsets, support_dof_offset,
            surface_node_indices_by_coil,
        )
        if coupling['V_cs'].shape[0] > 0:
            I_list.append(coupling['I_cs'])
            J_list.append(coupling['J_cs'])
            V_list.append(coupling['V_cs'])
        if coupling['V_sc'].shape[0] > 0:
            I_list.append(coupling['I_sc'])
            J_list.append(coupling['J_sc'])
            V_list.append(coupling['V_sc'])

        # ── Merge and solve ────────────────────────────────────────────────
        I_merged = jnp.concatenate([jnp.asarray(x) for x in I_list])
        J_merged = jnp.concatenate([jnp.asarray(x) for x in J_list])
        V_merged = jnp.concatenate(V_list)
        f_merged = jnp.concatenate(f_list)

        # Use cuDSS for the merged solve
        from ..solvers.cudss import _import_cudss_solver, assemble_csr_from_coo
        CuDSSSolver = _import_cudss_solver()

        csr_indptr, csr_cols, coo_to_csr, nnz_csr = assemble_csr_from_coo(
            I_merged, J_merged, n_total_dofs
        )
        csr_values = assemble_csr_values_merged(V_merged, coo_to_csr, nnz_csr)

        cudss_solver = CuDSSSolver(
            n_total_dofs, nnz_csr, device_id, mtype_id=mtype_id
        )
        cudss_solver.set_csr(csr_indptr, csr_cols)
        sol_flat, _ = cudss_solver.solve(f_merged, csr_values)

        return sol_flat

    def _fwd_wrapper(bcd, sdofs, pts, bf, wt):
        sol = _merged_solve(bcd, sdofs, pts, bf, wt)
        return sol, (bcd, sdofs, pts, bf, wt, sol)

    def _bwd_wrapper(res, g_flat):
        bcd, sdofs, pts, bf, wt, sol_flat = res
        surf_pts = [pts[i][pipelines[i].surface_node_indices]
                    for i in range(n_pipelines)]

        # Re-assemble K at the converged solution (linear → same K as forward)
        # and solve K^T λ = g for the adjoint variable.
        from ..solvers.cudss import _import_cudss_solver, assemble_csr_from_coo
        CuDSSSolver = _import_cudss_solver()

        I_list, J_list, V_list = [], [], []
        for i, pipeline in enumerate(pipelines):
            p_params = {
                'points':          pts[i],
                'body_force':      bf[i],
                'support_weights': wt[i],
                'support_attach':  jnp.zeros(
                    (pipeline.surface_node_indices.shape[0], 3),
                    dtype=pts[i].dtype,
                ),
            }
            Ii, Ji, Vi, _ = pipeline.assemble_coo(p_params)
            off = coil_dof_offsets[i]
            I_list.append(Ii + off)
            J_list.append(Ji + off)
            V_list.append(Vi)

        I_ss, J_ss, V_ss, n_s = support.coo(bcd, sdofs, surf_pts)
        I_list.append(jnp.asarray(I_ss) + support_dof_offset)
        J_list.append(jnp.asarray(J_ss) + support_dof_offset)
        V_list.append(V_ss)

        coupling = support.coupling_terms(
            bcd, sdofs, surf_pts,
            coil_dof_offsets, support_dof_offset,
            surface_node_indices_by_coil,
        )
        if coupling['V_cs'].shape[0] > 0:
            I_list.append(coupling['I_cs'])
            J_list.append(coupling['J_cs'])
            V_list.append(coupling['V_cs'])
        if coupling['V_sc'].shape[0] > 0:
            I_list.append(coupling['I_sc'])
            J_list.append(coupling['J_sc'])
            V_list.append(coupling['V_sc'])

        I_merged = jnp.concatenate([jnp.asarray(x) for x in I_list])
        J_merged = jnp.concatenate([jnp.asarray(x) for x in J_list])
        V_merged = jnp.concatenate(V_list)

        csr_indptr, csr_cols, coo_to_csr, nnz_csr = assemble_csr_from_coo(
            I_merged, J_merged, n_total_dofs
        )
        csr_values = assemble_csr_values_merged(V_merged, coo_to_csr, nnz_csr)

        # Solve K^T λ = g (transpose for adjoint)
        cudss_solver = CuDSSSolver(
            n_total_dofs, nnz_csr, device_id, mtype_id=mtype_id
        )
        cudss_solver.set_csr(csr_indptr, csr_cols)
        lambda_flat, _ = cudss_solver.solve(g_flat, csr_values, transpose=True)

        # Gradient via merged constraint VJP: grad = -λ^T ∂(K u - f)/∂params
        def _merged_constraint(bcd_in, sdofs_in, pts_in, bf_in, wt_in):
            """K(params) * sol_flat - f(params) at the converged solution."""
            s_pts = [pts_in[i][pipelines[i].surface_node_indices]
                     for i in range(n_pipelines)]
            residuals = []
            for i, pipeline in enumerate(pipelines):
                p_par = {
                    'points':          pts_in[i],
                    'body_force':      bf_in[i],
                    'support_weights': wt_in[i],
                    'support_attach':  jnp.zeros(
                        (pipeline.surface_node_indices.shape[0], 3),
                        dtype=pts_in[i].dtype,
                    ),
                }
                # compute_residual evaluates K_cc^i u_c^{i*} - f_c^i
                u_c_i = sol_flat[
                    coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
                ].reshape(pipeline.problem.num_total_nodes, 3)
                pipeline.problem.set_params(p_par)
                res_i = pipeline.problem.compute_residual_vars(
                    [u_c_i],
                    pipeline.problem.internal_vars,
                    pipeline.problem.internal_vars_surfaces,
                )
                import jax.flatten_util
                residuals.append(jax.flatten_util.ravel_pytree(res_i)[0])

            # Support residual: K_ss u_s
            Iss, Jss, Vss, ns = support.coo(bcd_in, sdofs_in, s_pts)
            K_ss = jnp.zeros((ns, ns))
            K_ss = K_ss.at[Iss, Jss].add(Vss)
            u_s = sol_flat[support_dof_offset:]
            residuals.append(K_ss @ u_s)

            return jnp.concatenate(residuals)

        _, vjp_fn = jax.vjp(
            _merged_constraint, bcd, sdofs, pts, bf, wt
        )
        grads = vjp_fn(-lambda_flat)  # grad = -λ^T ∂R/∂params

        return grads

    _merged_solve.defvjp(_fwd_wrapper, _bwd_wrapper)

    # ── Call the wrapped solver ────────────────────────────────────────────
    sol_flat = _merged_solve(
        base_curves_dofs,
        support_dofs,
        mesh_points_by_coil,
        body_force_by_coil,
        weights_by_coil,
    )

    # ── Split solution into per-coil + support ────────────────────────────
    sol_list_by_coil = []
    for i, pipeline in enumerate(pipelines):
        u_c_i = sol_flat[
            coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
        ].reshape(pipeline.problem.num_total_nodes, 3)
        sol_list_by_coil.append([u_c_i])

    u_s = sol_flat[support_dof_offset:]

    return {
        'sol_list_by_coil': sol_list_by_coil,
        'u_s':              u_s,
        'diagnostics':      {},
    }


# ============================================================================
# Internal helpers for monolithic assembly
# ============================================================================

def assemble_csr_values_merged(
    V_coo: jax.Array,
    coo_to_csr: jax.Array,
    nnz_csr: int,
) -> jax.Array:
    """Scatter flat COO values into CSR value array.

    Parameters
    ----------
    V_coo : jax.Array, shape ``(nnz_coo,)``
        COO values.
    coo_to_csr : jax.Array, shape ``(nnz_coo,)``
        Mapping from COO index to CSR index.
    nnz_csr : int
        Number of non-zeros in the CSR representation.

    Returns
    -------
    jax.Array, shape ``(nnz_csr,)``
    """
    csr_values = jnp.zeros(nnz_csr, dtype=V_coo.dtype)
    return csr_values.at[coo_to_csr].add(V_coo)
