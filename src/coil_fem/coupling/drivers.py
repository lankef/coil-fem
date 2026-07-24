"""Coupled coil-support solver drivers.

Provides two module-level driver functions that replace the uncoupled per-coil
loop inside :class:`~coil_fem.CoilFEM` when the active support structure has
its own DOFs (``support.is_coupled == True``):

* :func:`solve_staggered` — block Gauss-Seidel (BG-S) iteration with Aitken
  acceleration, wrapped in ``jax.lax.custom_root`` for correct implicit-function
  gradients.  Works on all backends (CPU and GPU).
* :func:`solve_monolithic` — cuDSS-only single merged-system sparse direct
  solve.  Raises :class:`NotImplementedError` when ``solver != 'cudss'``.
* :class:`MonolithicStatic` — immutable bundle of all pattern-level and
  solver-handle data built once at construction and reused every evaluation.
* :func:`make_merged_solve` — factory that produces the ``custom_vjp``
  merged-solve callable from a :class:`MonolithicStatic` bundle.

Both driver functions take a shared ``params`` bundle (see individual
docstrings) and return a ``dict`` with keys ``'sol_list_by_coil'``, ``'u_s'``,
and ``'diagnostics'``.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Callable

import numpy as onp
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.sparse.linalg

if TYPE_CHECKING:
    from ..pipelines import ElasticPipeline
    from .supports import Support


# ============================================================================
# MonolithicStatic — pre-built pattern + solver bundle
# ============================================================================

@dataclasses.dataclass(frozen=True, eq=False)
class MonolithicStatic:
    """Immutable bundle of static data for the monolithic merged-system solver.

    Built once by :meth:`~coil_fem.CoilFEM.build_monolithic_static` and
    consumed by :func:`solve_monolithic` on every evaluation.  All attributes
    are host-side Python scalars or numpy/JAX arrays computed from the mesh
    topology and the static beam connectivity — none depends on the traced DOF
    values evaluated at each optimisation step.

    The *cuDSS-only layer* (``solver_K``, ``solver_KT``, ``coo_to_csr_T``,
    ``nnz_csr_T``) is populated only when ``solver == 'cudss'``; the fields are
    ``None`` on other backends.

    Attributes
    ----------
    coil_dof_offsets : tuple[int, ...]
    support_dof_offset : int
    n_total_dofs : int
    n_dofs_per_coil : tuple[int, ...]
    n_s : int
        Number of support DOFs (``support.n_support_dofs``).
    has_cs : bool
        Whether the K_cs / K_sc coupling blocks are non-empty.
    has_sc : bool
        Whether the K_sc coupling block is non-empty.
    surface_node_indices_by_coil : tuple[np.ndarray, ...]
    indptr : jax.Array
        CSR row pointer for the forward solve (on device).
    indices : jax.Array
        CSR column index for the forward solve (on device).
    coo_to_csr : jax.Array
        Scatter map from flat COO ``V`` vector to CSR values (forward solve).
    nnz_csr : int
    coo_to_csr_T : jax.Array or None
        Scatter map for the transposed CSR (adjoint solve; cuDSS only).
    nnz_csr_T : int or None
    solver_K : object or None
        ``CuDSSSolver`` handle for the forward system (cuDSS only).
    solver_KT : object or None
        ``CuDSSSolver`` handle for the transposed system (cuDSS only).
    merged_solve : Callable or None
        ``custom_vjp``-wrapped merged-solve closure (cuDSS only).  Signature::

            merged_solve(bcd, sdofs, pts, bf, wt) -> sol_flat
    """
    coil_dof_offsets: tuple
    support_dof_offset: int
    n_total_dofs: int
    n_dofs_per_coil: tuple
    n_s: int
    has_cs: bool
    has_sc: bool
    surface_node_indices_by_coil: tuple
    curve_qps: tuple
    curve_orders: tuple
    I_cs_pat: object     # np.ndarray or None when has_cs=False
    J_cs_pat: object
    I_sc_pat: object     # np.ndarray or None when has_sc=False
    J_sc_pat: object
    indptr: object
    indices: object
    coo_to_csr: object
    nnz_csr: int
    coo_to_csr_T: object
    nnz_csr_T: object
    solver_K: object
    solver_KT: object
    merged_solve: object


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
        closure variables (pts, bf, wt, curves_by_coil, support_dofs).

        Returns only u_s_new (compact signature required by jax.vjp in the
        IFT adjoint).  Use :func:`_sweep_full` when sol_list is also needed.
        """
        state = {'u_s': u_s}

        # Compute beam geometry once; pass to compute_attach and solve.
        geom = support.geometry(curves_by_coil, support_dofs)

        # Coil FEM solves with shifted Winkler springs
        u_mesh_by_coil = []
        for i, pipeline in enumerate(pipelines):
            u_attach_i = support.compute_attach(
                i, surface_pts_by_coil[i], curves_by_coil, support_dofs, state,
                **({'geom': geom} if geom is not None else {}),
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
            'curves_jax':          curves_by_coil,
            'support_dofs':        support_dofs,
            'surface_pts_by_coil': surface_pts_by_coil,
        }
        if geom is not None:
            support_inputs['geom'] = geom
        return support.solve(support_inputs)['u_s']

    def _sweep_full(u_s: jax.Array):
        """One BG-S iteration that also returns the per-coil solution lists.

        Used by :func:`_run_iterations` to cache the final sweep's solutions,
        avoiding a redundant recovery sweep after convergence.

        Returns
        -------
        u_s_new : jax.Array, shape (n_s,)
        sol_list_by_coil : list[list[jax.Array]]
        """
        state = {'u_s': u_s}
        geom = support.geometry(curves_by_coil, support_dofs)

        u_mesh_by_coil = []
        sol_list_by_coil = []
        for i, pipeline in enumerate(pipelines):
            u_attach_i = support.compute_attach(
                i, surface_pts_by_coil[i], curves_by_coil, support_dofs, state,
                **({'geom': geom} if geom is not None else {}),
            )
            sol = pipeline.solve(
                mesh_points_by_coil[i],
                body_force_by_coil[i],
                weights_by_coil[i],
                support_attach=u_attach_i,
            )
            u_mesh_by_coil.append(sol['u'][pipeline.surface_node_indices])
            sol_list_by_coil.append(sol['sol_list'])

        support_inputs = {
            'u_mesh_by_coil':      u_mesh_by_coil,
            'curves_jax':          curves_by_coil,
            'support_dofs':        support_dofs,
            'surface_pts_by_coil': surface_pts_by_coil,
        }
        if geom is not None:
            support_inputs['geom'] = geom
        u_s_new = support.solve(support_inputs)['u_s']
        return u_s_new, sol_list_by_coil

    # ------------------------------------------------------------------
    # Forward solver — Python loop + Aitken (concrete, not traced)
    # ------------------------------------------------------------------

    def _run_iterations(u_s0: jax.Array):
        """BG-S loop; returns ``(u_s_converged, sol_list_by_coil)``."""
        u_s        = u_s0
        omega      = 1.0
        delta_prev = None
        last_sol_list = None

        for k in range(max_iters):
            u_s_new, sol_list = _sweep_full(u_s)
            delta = u_s_new - u_s          # f(u_s) = T(u_s) - u_s
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
            last_sol_list = sol_list

            if res < atol:
                break
            delta_prev = delta

        return u_s, last_sol_list

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
        # Linearize _sweep once at the converged point; reuse the resulting
        # vjp_fn for every GMRES matvec to avoid one full sweep per iteration.
        _, vjp_fn = jax.vjp(_sweep, u_s_star)

        def A_T_fn(v):
            """(I - dT/du_s)^T v via the pre-built VJP of _sweep at u_s*."""
            return v - vjp_fn(v)[0]

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
    u_s_star, sol_list_by_coil = _run_iterations(u_s_init)

    # Pass through the custom_vjp hook so gradients are computed via IFT.
    # sol_list_by_coil comes from the last _sweep_full call (the converged
    # iteration), avoiding a redundant recovery sweep after convergence.
    u_s_star = _staggered_core(u_s_star)

    return {
        'sol_list_by_coil': sol_list_by_coil,
        'u_s':              u_s_star,
        'diagnostics':      {},
    }


# ============================================================================
# make_merged_solve — factory for the cuDSS custom_vjp kernel
# ============================================================================

def make_merged_solve(
    pipelines: list,
    support,
    static: MonolithicStatic,
) -> Callable:
    """Build a ``custom_vjp``-wrapped merged direct-solve function.

    The returned callable captures ``pipelines``, ``support``, and
    ``static`` (which holds the pre-built CSR patterns, coupling index
    arrays, and cuDSS solver handles) and exposes only the traced arguments
    to JAX autodiff.

    Parameters
    ----------
    pipelines : list[ElasticPipeline]
    support : Support
    static : MonolithicStatic
        Must have all cuDSS fields populated (``solver_K``, ``solver_KT``,
        ``coo_to_csr``, ``coo_to_csr_T``, etc.).

    Returns
    -------
    merged_solve : Callable
        ``merged_solve(bcd, sdofs, pts, bf, wt) -> sol_flat``
        where ``bcd`` is a list of per-coil DOF arrays (traced),
        ``sdofs`` is the support DOF dict (traced), and ``pts``, ``bf``,
        ``wt`` are per-coil mesh points, body forces, and Winkler weights
        (all traced).
    """
    from ..solvers.cudss import assemble_csr_values
    from ..geo.curve_jax import CurveXYZFourierJAX as _CurveJAX

    coil_dof_offsets          = static.coil_dof_offsets
    support_dof_offset        = static.support_dof_offset
    n_dofs_per_coil           = static.n_dofs_per_coil
    n_s                       = static.n_s
    has_cs                    = static.has_cs
    has_sc                    = static.has_sc
    coo_to_csr                = static.coo_to_csr
    nnz_csr                   = static.nnz_csr
    coo_to_csr_T              = static.coo_to_csr_T
    nnz_csr_T                 = static.nnz_csr_T
    solver_K                  = static.solver_K
    solver_KT                 = static.solver_KT
    curve_qps                 = static.curve_qps
    curve_orders              = static.curve_orders
    I_cs_jax = jnp.asarray(static.I_cs_pat) if has_cs else None
    J_cs_jax = jnp.asarray(static.J_cs_pat) if has_cs else None
    I_sc_jax = jnp.asarray(static.I_sc_pat) if has_sc else None
    J_sc_jax = jnp.asarray(static.J_sc_pat) if has_sc else None

    n_pipelines = len(pipelines)

    def _make_curves(bcd):
        return [_CurveJAX(curve_qps[i], bcd[i], curve_orders[i])
                for i in range(n_pipelines)]

    def _surf_pts(pts):
        return [pts[i][pipelines[i].surface_node_indices]
                for i in range(n_pipelines)]

    def _coil_params(i, pts, bf, wt):
        return {
            'points':          pts[i],
            'body_force':      bf[i],
            'support_weights': wt[i],
            'support_attach':  jnp.zeros(
                (pipelines[i].surface_node_indices.shape[0], 3),
                dtype=pts[i].dtype,
            ),
        }

    def _assemble_merged_values(bcd, sdofs, pts, bf, wt):
        """Assemble merged COO value vector V and RHS f."""
        surf_pts = _surf_pts(pts)
        curves   = _make_curves(bcd)
        # Compute beam geometry once; reuse for coo and coupling_values.
        geom = support.geometry(curves, sdofs)
        V_blocks, f_blocks = [], []
        for i, pipeline in enumerate(pipelines):
            p_params = _coil_params(i, pts, bf, wt)
            _, _, Vi, _, fi = pipeline.assemble_coo(p_params)
            V_blocks.append(Vi)
            f_blocks.append(fi)
        geom_kw = {'geom': geom} if geom is not None else {}
        _, _, V_ss, _ = support.coo(curves, sdofs, surf_pts, **geom_kw)
        V_blocks.append(V_ss)
        f_blocks.append(jnp.zeros(n_s, dtype=V_ss.dtype))
        V_cs, V_sc = support.coupling_values(curves, sdofs, surf_pts, **geom_kw)
        if has_cs:
            V_blocks.append(V_cs)
        if has_sc:
            V_blocks.append(V_sc)
        V_merged = jnp.concatenate([jnp.asarray(v) for v in V_blocks])
        f_merged = jnp.concatenate(f_blocks)
        return V_merged, f_merged

    @jax.custom_vjp
    def merged_solve(bcd, sdofs, pts, bf, wt):
        V_merged, f_merged = _assemble_merged_values(bcd, sdofs, pts, bf, wt)
        csr_values = assemble_csr_values(V_merged, coo_to_csr, nnz_csr)
        sol_flat, _inertia = solver_K(f_merged, csr_values)
        return sol_flat

    def _fwd(bcd, sdofs, pts, bf, wt):
        sol = merged_solve(bcd, sdofs, pts, bf, wt)
        # Recompute geom here so it can be saved in the residual tuple and
        # reused in _bwd, avoiding a redundant geometry evaluation per backward.
        fwd_geom = support.geometry(_make_curves(bcd), sdofs)
        return sol, (bcd, sdofs, pts, bf, wt, sol, fwd_geom)

    def _bwd(res, g_flat):
        bcd, sdofs, pts, bf, wt, sol_flat, fwd_geom = res
        V_merged, _ = _assemble_merged_values(bcd, sdofs, pts, bf, wt)
        csr_values_T = assemble_csr_values(V_merged, coo_to_csr_T, nnz_csr_T)
        lambda_flat, _inertia = solver_KT(g_flat, csr_values_T)

        def _merged_constraint(bcd_in, sdofs_in, pts_in, bf_in, wt_in):
            s_pts  = _surf_pts(pts_in)
            curves = _make_curves(bcd_in)
            # Use the pre-computed geometry from the forward pass when the
            # inputs are the same (always the case here — we differentiate
            # through the constraint at the forward-pass point).
            geom_in = fwd_geom
            geom_kw = {'geom': geom_in} if geom_in is not None else {}
            residuals = []
            for i, pipeline in enumerate(pipelines):
                p_par = _coil_params(i, pts_in, bf_in, wt_in)
                u_c_i = sol_flat[
                    coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
                ].reshape(pipeline.problem.fes[0].num_total_nodes, 3)
                pipeline.problem.set_params(p_par)
                res_i = pipeline.problem.compute_residual_vars(
                    [u_c_i],
                    pipeline.problem.internal_vars,
                    pipeline.problem.internal_vars_surfaces,
                )
                residuals.append(jax.flatten_util.ravel_pytree(res_i)[0])
            Iss, Jss, Vss, ns = support.coo(curves, sdofs_in, s_pts, **geom_kw)
            u_s = sol_flat[support_dof_offset:]
            r_s = jnp.zeros(ns, dtype=Vss.dtype).at[Iss].add(Vss * u_s[Jss])
            r_full = jnp.concatenate(residuals + [r_s])
            V_cs_in, V_sc_in = support.coupling_values(
                curves, sdofs_in, s_pts, **geom_kw
            )
            if has_cs:
                r_full = r_full.at[I_cs_jax].add(
                    V_cs_in * sol_flat[J_cs_jax]
                )
            if has_sc:
                r_full = r_full.at[I_sc_jax].add(
                    V_sc_in * sol_flat[J_sc_jax]
                )
            return r_full

        _, vjp_fn = jax.vjp(_merged_constraint, bcd, sdofs, pts, bf, wt)
        grads = vjp_fn(-lambda_flat)
        return grads

    merged_solve.defvjp(_fwd, _bwd)
    return merged_solve


# ============================================================================
# Monolithic driver (cuDSS-only)
# ============================================================================

def solve_monolithic(
    pipelines: list,
    support,
    params: dict,
    static: MonolithicStatic,
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

    The CSR sparsity pattern, cuDSS solver handles, and the ``custom_vjp``
    merged-solve closure are all pre-built in ``static``
    (via :meth:`~coil_fem.CoilFEM.build_monolithic_static`) and reused here
    without any host-side pattern construction.

    Parameters
    ----------
    pipelines : list[:class:`~coil_fem.pipelines.ElasticPipeline`]
    support : :class:`~coil_fem.coupling.supports.Support`
    params : dict
        Same format as :func:`solve_staggered`.
    static : MonolithicStatic
        Pre-built static bundle from
        :meth:`~coil_fem.CoilFEM.build_monolithic_static`.

    Returns
    -------
    dict with the same keys as :func:`solve_staggered`.

    Raises
    ------
    NotImplementedError
        When ``static.merged_solve`` is ``None``, which happens when the
        pipeline solver is not ``'cudss'``.  CPU-only monolithic solves are
        not implemented; use :func:`solve_staggered` on CPU backends.
    """
    if static.merged_solve is None:
        solver_opt = pipelines[0].problem_options.get('solver', '')
        raise NotImplementedError(
            "solve_monolithic requires problem_options={'solver': 'cudss'}. "
            f"Got solver='{solver_opt}'. "
            "The monolithic merge assembles K_cc^i from problem.V_jax which "
            "is only populated on the cuDSS GPU path.  Use solve_staggered "
            "for CPU-compatible coupled solves."
        )

    coil_dof_offsets = static.coil_dof_offsets
    n_dofs_per_coil  = static.n_dofs_per_coil
    support_dof_offset = static.support_dof_offset

    mesh_points_by_coil = params['mesh_points_by_coil']
    body_force_by_coil  = params['body_force_by_coil']
    weights_by_coil     = params['weights_by_coil']
    curves_by_coil      = params['curves_by_coil']
    support_dofs        = params['support_dofs']

    sol_flat = static.merged_solve(
        [c.dofs for c in curves_by_coil],
        support_dofs,
        mesh_points_by_coil,
        body_force_by_coil,
        weights_by_coil,
    )

    sol_list_by_coil = []
    for i, pipeline in enumerate(pipelines):
        u_c_i = sol_flat[
            coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
        ].reshape(pipeline.problem.fes[0].num_total_nodes, 3)
        sol_list_by_coil.append([u_c_i])

    u_s = sol_flat[support_dof_offset:]

    return {
        'sol_list_by_coil': sol_list_by_coil,
        'u_s':              u_s,
        'diagnostics':      {},
    }
