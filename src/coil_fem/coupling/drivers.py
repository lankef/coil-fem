"""Coupled coil-support solver drivers.

Provides driver functions and a static-bundle dataclass used when
``support.is_coupled == True``:

* :func:`solve_staggered` — **retired**; raises :class:`NotImplementedError`.
  See ``notes/PLANS.md`` for the analysis.
* :func:`solve_monolithic` — cuDSS-only single merged-system sparse direct
  solve.  Raises :class:`NotImplementedError` when ``solver != 'cudss'``.
* :class:`MonolithicStatic` — immutable bundle of all pattern-level and
  solver-handle data built once at construction and reused every evaluation.
* :func:`make_merged_solve` — factory that produces the ``custom_vjp``
  merged-solve callable from a :class:`MonolithicStatic` bundle.

Both active driver functions take a shared ``params`` bundle (see individual
docstrings) and return a ``dict`` with keys ``'sol_list_by_coil'``, ``'u_s'``,
and ``'diagnostics'``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import numpy as onp
import jax
import jax.flatten_util
import jax.numpy as jnp

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

    The *cuDSS-only layer* (``solver_K``, and optionally ``solver_KT`` /
    ``coo_to_csr_T`` / ``nnz_csr_T``) is populated only when
    ``solver == 'cudss'``; the fields are ``None`` on other backends.
    ``solver_KT`` / ``coo_to_csr_T`` are built only when
    ``adjoint_reuses_K`` is ``False`` (merged matrix claim is general).

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
    I_ss_pat, J_ss_pat : np.ndarray
        Global COO indices for the support ``K_ss`` block (from
        ``support.support_pattern`` shifted by ``support_dof_offset``).
    I_cs_pat, J_cs_pat, I_sc_pat, J_sc_pat : np.ndarray or None
        Global COO indices for the off-diagonal coupling blocks.
    indptr : jax.Array
        CSR row pointer for the forward solve (on device).
    indices : jax.Array
        CSR column index for the forward solve (on device).
    coo_to_csr : jax.Array
        Scatter map from flat COO ``V`` vector to CSR values (forward solve).
    nnz_csr : int
    adjoint_reuses_K : bool
        When ``True``, the adjoint solves ``K λ = g`` with ``solver_K`` and
        the forward CSR (valid for symmetric / SPD merged ``K``).  When
        ``False``, a separate transposed CSR / ``solver_KT`` is used.
    coo_to_csr_T : jax.Array or None
        Scatter map for the transposed CSR (adjoint; only when
        ``adjoint_reuses_K`` is ``False``).
    nnz_csr_T : int or None
    solver_K : object or None
        ``CuDSSSolver`` handle for the forward system (cuDSS only).
    solver_KT : object or None
        ``CuDSSSolver`` handle for the transposed system (cuDSS only; only
        when ``adjoint_reuses_K`` is ``False``).
    merged_solve : Callable or None
        ``custom_vjp``-wrapped merged-solve closure (cuDSS only).  Signature::

            merged_solve(bcd, sdofs, pts, bf, k, fe_geom, geom) -> sol_flat
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
    I_ss_pat: object     # np.ndarray — global COO row indices for K_ss
    J_ss_pat: object     # np.ndarray — global COO column indices for K_ss
    I_cs_pat: object     # np.ndarray or None when has_cs=False
    J_cs_pat: object
    I_sc_pat: object     # np.ndarray or None when has_sc=False
    J_sc_pat: object
    indptr: object
    indices: object
    coo_to_csr: object
    nnz_csr: int
    adjoint_reuses_K: bool
    coo_to_csr_T: object
    nnz_csr_T: object
    solver_K: object
    solver_KT: object
    merged_solve: object


# ============================================================================
# Uncoupled driver (independent per-coil loop)
# ============================================================================

def solve_uncoupled(
    pipelines: list,
    support,
    params: dict,
) -> dict:
    """Independent per-coil FEM solves; no support DOFs.

    Each pipeline is solved separately using its own ``params`` slice.
    The support is not involved beyond providing the Winkler spring
    stiffness already baked into ``params['stiffness_by_coil']``.

    Parameters
    ----------
    pipelines : list[:class:`~coil_fem.pipelines.ElasticPipeline`]
        One per base coil.
    support : :class:`~coil_fem.coupling.supports.Support`
        Accepted for API symmetry with the other drivers; not called here.
    params : dict
        Required keys:

        * ``'mesh_points_by_coil'``  : list[jax.Array]
        * ``'body_force_by_coil'``   : list[jax.Array]
        * ``'stiffness_by_coil'``    : list[jax.Array]

    Returns
    -------
    dict
        Same keys as :func:`solve_monolithic`:
        ``'sol_list_by_coil'``, ``'u_s'`` (``None``), ``'diagnostics'`` (``{}``).
    """
    pts_by_coil    = params['mesh_points_by_coil']
    bf_by_coil     = params['body_force_by_coil']
    k_by_coil      = params['stiffness_by_coil']
    fe_geom_by_coil = params.get('fe_geom_by_coil')
    sol_list_by_coil = [
        pipelines[i].solve(
            pts_by_coil[i], bf_by_coil[i], k_by_coil[i],
            fe_geom=fe_geom_by_coil[i] if fe_geom_by_coil is not None else None,
        )['sol_list']
        for i in range(len(pipelines))
    ]
    return {
        'sol_list_by_coil': sol_list_by_coil,
        'u_s':              None,
        'diagnostics':      {},
    }


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
        * ``'stiffness_by_coil'``    : list[jax.Array]
          Per-surface-quad Winkler stiffness per coil [N/m³], shape ``(n_sq_i,)``.
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
    raise NotImplementedError(
        "solve_staggered is numerically unsound and has been retired.\n"
        "Use coupling='monolithic' with problem_options={'solver': 'cudss'} "
        "for coupled coil-support solves.\n"
        "See notes/PLANS.md — 'Staggered coupling is numerically unsound' — "
        "for the full analysis."
    )


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
        Must have cuDSS fields populated (``solver_K``, ``coo_to_csr``,
        etc.).  ``solver_KT`` / ``coo_to_csr_T`` are required only when
        ``static.adjoint_reuses_K`` is ``False``.

    Returns
    -------
    merged_solve : Callable
        ``merged_solve(bcd, sdofs, pts, bf, k, fe_geom, geom) -> sol_flat``
        where ``bcd`` is a list of per-coil DOF arrays (traced),
        ``sdofs`` is the support DOF dict (traced), ``pts`` / ``bf`` / ``k``
        are per-coil mesh points, body forces, and Winkler stiffness,
        ``fe_geom`` is a per-coil ``(sg, jxw, vgj, pqp)`` list from
        :func:`~coil_fem.problems.recompute_fe_geometry`, and ``geom`` is
        the precomputed :meth:`~coil_fem.coupling.Support.beam_geometry`
        dict (or ``None``).
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
    adjoint_reuses_K          = static.adjoint_reuses_K
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

    # Static interp maps per coil for folding quad-pt weights back to node DOFs
    # in coupling_values (K_cs / K_sc blocks).
    surf_interp_by_coil = [
        (
            pipelines[i].problem._sel_face_sv,
            pipelines[i].problem._surf_face_to_surf_node,
            int(pipelines[i].problem._surf_unique_global_nodes.shape[0]),
        )
        for i in range(n_pipelines)
    ]

    def _make_curves(bcd):
        return [_CurveJAX(curve_qps[i], bcd[i], curve_orders[i])
                for i in range(n_pipelines)]

    def _surf_quad_pts(pts):
        return [pipelines[i].surface_quad_points(pts[i])
                for i in range(n_pipelines)]

    def _surf_jxw(pts):
        return [pipelines[i].problem.surface_jxw(pts[i])
                for i in range(n_pipelines)]

    def _coil_params(i, pts, bf, k, fe_geom=None):
        p = {
            'points':      pts[i],
            'body_force':  bf[i],
            'support_k':   k[i],
        }
        if fe_geom is not None:
            p['_fe_geom'] = fe_geom[i]
        return p

    def _assemble_merged_values(bcd, sdofs, pts, bf, k, fe_geom, geom):
        """Assemble merged COO value vector V, RHS f, and beam geometry.

        ``fe_geom`` / ``geom`` are precomputed by the caller (``_solve_all``)
        so volume FE geometry and ``beam_geometry`` are not recomputed on the
        *forward* path.  The custom_vjp backward constraint must still
        recompute ``beam_geometry`` from ``sdofs`` (see ``_bwd``) — do not
        treat this forward cache as an adjoint freeze.
        Returns ``geom`` (possibly computed as a fallback).
        """
        surf_pts = _surf_quad_pts(pts)
        jxw_list = _surf_jxw(pts)
        curves   = _make_curves(bcd)
        if geom is None:
            geom = support.beam_geometry(curves, sdofs)
        V_blocks, f_blocks = [], []
        for i, pipeline in enumerate(pipelines):
            p_params = _coil_params(i, pts, bf, k, fe_geom)
            _, _, Vi, _, fi = pipeline.assemble_coo(p_params)
            V_blocks.append(Vi)
            f_blocks.append(fi)
        geom_kw = {'geom': geom} if geom is not None else {}
        V_ss = support.support_values(
            curves, sdofs, surf_pts, **geom_kw, jxw_by_coil=jxw_list,
        )
        V_blocks.append(V_ss)
        f_blocks.append(jnp.zeros(n_s, dtype=V_ss.dtype))
        V_cs, V_sc = support.coupling_values(
            curves, sdofs, surf_pts,
            surf_interp_by_coil=surf_interp_by_coil,
            jxw_by_coil=jxw_list,
            **geom_kw,
        )
        if has_cs:
            V_blocks.append(V_cs)
        if has_sc:
            V_blocks.append(V_sc)
        V_merged = jnp.concatenate([jnp.asarray(v) for v in V_blocks])
        f_merged = jnp.concatenate(f_blocks)
        return V_merged, f_merged, geom

    @jax.custom_vjp
    def merged_solve(bcd, sdofs, pts, bf, k, fe_geom, geom):
        V_merged, f_merged, _ = _assemble_merged_values(
            bcd, sdofs, pts, bf, k, fe_geom, geom,
        )
        csr_values = assemble_csr_values(V_merged, coo_to_csr, nnz_csr)
        # inertia = (n_pos, n_neg, n_zero); verified positive-definite when
        # n_neg == n_zero == 0.  Threading through custom_vjp return would
        # require a paired _fwd / _bwd update — kept as a local here.
        sol_flat, inertia = solver_K(f_merged, csr_values)
        return sol_flat

    def _fwd(bcd, sdofs, pts, bf, k, fe_geom, geom):
        # Call _assemble_merged_values directly (rather than through merged_solve)
        # to capture V_merged for the backward pass.  Stashing V_merged avoids a
        # full Jacobian assembly in _bwd (review item 4g).
        # ``geom`` is kept in the residual only to build a zero cotangent for the
        # outer forward-cache argument — it must NOT be reused inside the
        # constraint VJP (see _bwd).
        V_merged, f_merged, _ = _assemble_merged_values(
            bcd, sdofs, pts, bf, k, fe_geom, geom,
        )
        csr_values = assemble_csr_values(V_merged, coo_to_csr, nnz_csr)
        sol_flat, _inertia = solver_K(f_merged, csr_values)
        return sol_flat, (
            bcd, sdofs, pts, bf, k, fe_geom, sol_flat, geom, V_merged,
        )

    def _bwd(res, g_flat):
        bcd, sdofs, pts, bf, k, fe_geom, sol_flat, geom_arg, V_merged = res
        # Reuse V_merged from the forward pass — same system matrix, no need to
        # reassemble.  Symmetric / SPD: Kᵀ = K → reuse solver_K + forward CSR.
        if adjoint_reuses_K:
            csr_values = assemble_csr_values(V_merged, coo_to_csr, nnz_csr)
            lambda_flat, _inertia = solver_K(g_flat, csr_values)
        else:
            csr_values_T = assemble_csr_values(V_merged, coo_to_csr_T, nnz_csr_T)
            lambda_flat, _inertia = solver_KT(g_flat, csr_values_T)

        def _merged_constraint(bcd_in, sdofs_in, pts_in, bf_in, k_in, fe_geom_in):
            s_quad_pts = _surf_quad_pts(pts_in)
            jxw_list_in = _surf_jxw(pts_in)
            curves = _make_curves(bcd_in)
            # ------------------------------------------------------------------
            # REQUIRED FOR CORRECT GRADIENTS — do not "optimize" this away.
            #
            # SupportBeams attachment DOFs (phis_*, x_foundation, thetas_*)
            # enter K_ss / K_cs / K_sc almost entirely through beam_geometry
            # (endpoints, gamma3, L_eff).  Reusing the forward-cached geom_arg
            # here freezes that path, so jax.grad / CoilFEMObjective.dJ miss
            # ∂K/∂φ and Taylor tests / L-BFGS on beam networks fail badly
            # (analytic |dJh| ≫ FD).  Forward-only sharing of support_geom in
            # _solve_all / _assemble_merged_values is fine; the constraint VJP
            # must recompute.  For memory, use jax.checkpoint/remat — never
            # freeze geom.
            #
            # Grounded Winkler k (incl. k_att*w_a) remains an outer argument;
            # its support-DOF grads flow via g_k through the producer in
            # _solve_all.
            # ------------------------------------------------------------------
            geom_in = support.beam_geometry(curves, sdofs_in)
            geom_kw = {'geom': geom_in} if geom_in is not None else {}
            residuals = []
            for i, pipeline in enumerate(pipelines):
                p_par = _coil_params(i, pts_in, bf_in, k_in, fe_geom_in)
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
            Iss, Jss = support.support_pattern()
            Vss = support.support_values(
                curves, sdofs_in, s_quad_pts,
                jxw_by_coil=jxw_list_in,
                **geom_kw,
            )
            u_s = sol_flat[support_dof_offset:]
            ns = n_s
            r_s = jnp.zeros(ns, dtype=Vss.dtype).at[Iss].add(Vss * u_s[Jss])
            r_full = jnp.concatenate(residuals + [r_s])
            V_cs_in, V_sc_in = support.coupling_values(
                curves, sdofs_in, s_quad_pts,
                surf_interp_by_coil=surf_interp_by_coil,
                jxw_by_coil=jxw_list_in,
                **geom_kw,
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

        _, vjp_fn = jax.vjp(
            _merged_constraint, bcd, sdofs, pts, bf, k, fe_geom,
        )
        g_bcd, g_sdofs, g_pts, g_bf, g_k, g_fe_geom = vjp_fn(-lambda_flat)
        # Outer ``geom`` is a forward-only cache from _solve_all.  Support-DOF
        # derivatives for coupling flow through sdofs → beam_geometry inside
        # the constraint above — not through this cotangent.  Always zero.
        g_geom = jax.tree_util.tree_map(
            jnp.zeros_like, geom_arg,
        ) if geom_arg is not None else None
        return g_bcd, g_sdofs, g_pts, g_bf, g_k, g_fe_geom, g_geom

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
        Required keys (from :meth:`~coil_fem.CoilFEM._solve_all`):

        * ``'mesh_points_by_coil'``, ``'body_force_by_coil'``,
          ``'stiffness_by_coil'``, ``'curves_by_coil'``, ``'support_dofs'``
        * ``'fe_geom_by_coil'`` — per-coil ``(sg, jxw, vgj, pqp)``
        * ``'support_geom'`` — precomputed beam geometry dict (or ``None``)
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

    # Catch Host-pinned JAX before spineax's CUDA-only FFI fails opaquely.
    from ..solvers.cudss import require_cuda_for_cudss
    require_cuda_for_cudss()

    coil_dof_offsets = static.coil_dof_offsets
    n_dofs_per_coil  = static.n_dofs_per_coil
    support_dof_offset = static.support_dof_offset

    mesh_points_by_coil = params['mesh_points_by_coil']
    body_force_by_coil  = params['body_force_by_coil']
    stiffness_by_coil   = params['stiffness_by_coil']
    curves_by_coil      = params['curves_by_coil']
    support_dofs        = params['support_dofs']
    fe_geom_by_coil     = params['fe_geom_by_coil']
    support_geom        = params.get('support_geom')

    sol_flat = static.merged_solve(
        [c.dofs for c in curves_by_coil],
        support_dofs,
        mesh_points_by_coil,
        body_force_by_coil,
        stiffness_by_coil,
        fe_geom_by_coil,
        support_geom,
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
