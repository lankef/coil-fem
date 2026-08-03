"""GPU sparse direct solver for JAX-FEM using spineax / NVIDIA cuDSS.

Provides :class:`CuDSSNewtonSolver` and :func:`cudss_ad_wrapper`, a drop-in
replacement for ``jax_fem.solver.ad_wrapper`` that keeps the Jacobian on the
GPU device throughout the Newton loop and reuses the cuDSS factorisation for
both the forward and adjoint solves.

Requires an NVIDIA GPU, ``spineax`` (see ``.[cudss]`` install extra), and
``jax_enable_x64 = True``.  ``spineax`` is an optional dependency: this module
imports cleanly without it; the solver raises a helpful error only when actually
constructed.
"""

from __future__ import annotations

import functools
import importlib.util
import time
import warnings
from typing import TYPE_CHECKING

import numpy as onp
import jax
import jax.numpy as jnp
import jax.flatten_util

from jax_fem import logger
from jax_fem.solver import (
    apply_bc_vec,
    get_flatten_fn,
    apply_bc,
)

_SPINEAX_INSTALL_HINT = (
    "coil_fem.solvers.cudss requires the optional GPU stack (spineax + "
    "NVIDIA cuDSS), which is not installed. Install it with:\n"
    "  conda install -c conda-forge cuda-nvcc=<version>\n"
    '  pip install --no-build-isolation -e ".[cudss]"'
)

_CUDA_BACKEND_HINT = (
    "problem_options['solver']='cudss' requires a CUDA JAX backend, but "
    "jax.default_backend()={backend!r} and jax.devices()={devices}. "
    "simsopt sets jax_platform_name='cpu' on import, which pins JAX to Host "
    "if it initialises before a GPU backend is selected. Fix: restart the "
    "process/kernel and set os.environ['JAX_PLATFORMS']='cuda' before "
    "importing simsopt (or import coil_fem before simsopt)."
)

# ============================================================================
# Matrix symmetry helpers
# ============================================================================

# Maps solver-agnostic symmetry string → cuDSS mtype_id integer.
# mview_id is always 0 (CUDSS_MVIEW_FULL): we always supply the full matrix.
_MTYPE_ID: dict[str, int] = {'general': 0, 'symmetric': 1, 'spd': 3}
# Ordering for weakest-claim meet: lower = weaker assumption.
_STRENGTH: dict[str, int] = {'general': 0, 'symmetric': 1, 'spd': 2}


def weakest_symmetry(*claims: str) -> str:
    """Weakest (least-assuming) symmetry claim among ``claims``.

    Parameters
    ----------
    *claims : str
        One or more strings from ``{'general', 'symmetric', 'spd'}``.

    Returns
    -------
    str
        The claim with the smallest entry in ``_STRENGTH``.
    """
    return min(claims, key=_STRENGTH.__getitem__)


def adjoint_reuses_forward_K(merged_sym: str, mtype_id: int) -> bool:
    """True when ``Kᵀ = K`` so the adjoint may reuse ``solver_K`` + forward CSR.

    Honours the *final* ``mtype_id`` (after any ``cudss_mtype_id`` override):
    ``0`` (general) → ``False``; ``1`` (symmetric) and ``3`` (SPD) → ``True``.
    ``merged_sym`` is accepted for call-site clarity / future checks but the
    decision is driven by ``mtype_id``.

    Parameters
    ----------
    merged_sym : str
        Weakest merged claim from :func:`weakest_symmetry`.
    mtype_id : int
        Final cuDSS matrix type id (0=general, 1=symmetric, 3=SPD).

    Returns
    -------
    bool
    """
    if merged_sym not in _MTYPE_ID:
        raise ValueError(
            f"merged_sym must be one of {sorted(_MTYPE_ID)}, got {merged_sym!r}"
        )
    return mtype_id != 0


def require_cuda_for_cudss() -> None:
    """Raise if the cuDSS path is used without a CUDA JAX device.

    spineax registers ``solve_single_f64`` only for ``platform='CUDA'``.  When
    JAX is pinned to Host (commonly by simsopt's
    ``jax_platform_name='cpu'``), the FFI call fails with an opaque
    ``NOT_FOUND`` error; this check fails earlier with a clear fix.
    """
    devices = jax.devices()
    if not any(d.platform == 'gpu' for d in devices):
        raise RuntimeError(
            _CUDA_BACKEND_HINT.format(
                backend=jax.default_backend(),
                devices=devices,
            )
        )


def _import_cudss_solver():
    """Import spineax's ``CuDSSSolver``, raising a helpful error if missing."""
    require_cuda_for_cudss()
    try:
        from spineax.cudss.solver import CuDSSSolver
    except ImportError as e:
        raise ImportError(_SPINEAX_INSTALL_HINT) from e
    return CuDSSSolver

if TYPE_CHECKING:
    from jax_fem.problem import Problem
    from ..problems import LinearElasticity3D


# ============================================================================
# Build CSR pattern from COO triplets (host-side, run once per problem)
# ============================================================================

def build_csr_pattern(I: onp.ndarray, J: onp.ndarray, n: int):
    """Convert the COO sparsity pattern of a JAX-FEM problem to CSR + metadata.

    Parameters
    ----------
    I, J : numpy.ndarray (nnz_coo,)
        Row and column indices from ``problem.I`` and ``problem.J``.
        May contain duplicate (i, j) pairs whose Jacobian values are summed.
    n : int
        Matrix dimension — ``problem.num_total_dofs_all_vars``.

    Returns
    -------
    indptr : jnp.ndarray (n+1,) int32
        CSR row-pointer array — uploaded to device.
    indices : jnp.ndarray (nnz_csr,) int32
        CSR column-index array (sorted within each row) — on device.
    coo_to_csr : jnp.ndarray (nnz_coo,) int32
        Maps each COO entry to its CSR slot:
        ``csr_values[coo_to_csr[k]] += V_flat[k]``.
    row_per_nnz : jnp.ndarray (nnz_csr,) int32
        Row index for every non-zero slot in the CSR matrix.
    diag_slots : jnp.ndarray (n,) int32
        CSR value index of the diagonal entry A[i, i] for each row i.
    nnz_csr : int
        Number of unique non-zero entries after deduplication.
    """
    I = onp.asarray(I, dtype=onp.int64)
    J = onp.asarray(J, dtype=onp.int64)

    # --- 1. Lex-sort (row, col) and deduplicate to get unique (i,j) pairs ---
    sort_idx = onp.lexsort((J, I))
    I_s = I[sort_idx]
    J_s = J[sort_idx]

    # Pack into a single key so that onp.unique deduplicates (i, j) pairs.
    # Safe for n < 2^31 (billions of DOFs would be unusual).
    key = I_s.astype(onp.int64) * (n + 1) + J_s.astype(onp.int64)
    _, first_occ, inv = onp.unique(key, return_index=True, return_inverse=True)
    nnz_csr = int(len(first_occ))

    # --- 2. CSR arrays ---------------------------------------------------------
    indices_np = J_s[first_occ].astype(onp.int32)
    rows_unique = I_s[first_occ].astype(onp.int32)

    row_counts = onp.bincount(rows_unique, minlength=n).astype(onp.int32)
    indptr_np = onp.zeros(n + 1, dtype=onp.int32)
    onp.cumsum(row_counts, out=indptr_np[1:])

    # --- 3. row_per_nnz: row index for every CSR slot -------------------------
    row_per_nnz_np = onp.repeat(onp.arange(n, dtype=onp.int32), row_counts)

    # --- 4. coo_to_csr: map each original COO entry → CSR slot ---------------
    coo_to_csr_np = onp.empty(len(I), dtype=onp.int32)
    coo_to_csr_np[sort_idx] = inv.astype(onp.int32)

    # --- 5. Diagonal slots: CSR slot k where row_per_nnz[k] == indices[k] ----
    is_diag = indices_np == row_per_nnz_np
    diag_slot_indices = onp.where(is_diag)[0]           # CSR slot indices of diag entries
    diag_rows = row_per_nnz_np[diag_slot_indices]        # which rows they belong to

    if len(diag_slot_indices) != n:
        missing = set(range(n)) - set(diag_rows.tolist())
        raise ValueError(
            f"build_csr_pattern: diagonal entries missing for {len(missing)} DOF(s). "
            f"Example missing: {sorted(missing)[:5]}. "
            "Every FEM DOF must appear in at least one element."
        )

    diag_slots_np = onp.empty(n, dtype=onp.int32)
    diag_slots_np[diag_rows] = diag_slot_indices.astype(onp.int32)

    return (
        jnp.asarray(indptr_np),
        jnp.asarray(indices_np),
        jnp.asarray(coo_to_csr_np),
        jnp.asarray(row_per_nnz_np),
        jnp.asarray(diag_slots_np),
        nnz_csr,
    )


# ============================================================================
# On-device assembly and BC application
# ============================================================================

@functools.partial(jax.jit, static_argnums=(2,))
def assemble_csr_values(
    V_flat: jnp.ndarray,
    coo_to_csr: jnp.ndarray,
    nnz_csr: int,
) -> jnp.ndarray:
    """Scatter-add COO Jacobian values into CSR slots, on device.

    Parameters
    ----------
    V_flat : jnp.ndarray (nnz_coo,)
        Flat Jacobian values from ``problem.V_jax``.
    coo_to_csr : jnp.ndarray (nnz_coo,) int32
        Pre-computed scatter map from ``build_csr_pattern``.
    nnz_csr : int
        Number of unique CSR non-zeros (static).

    Returns
    -------
    jnp.ndarray (nnz_csr,)
        CSR values with duplicate contributions summed.
    """
    return jnp.zeros(nnz_csr, dtype=V_flat.dtype).at[coo_to_csr].add(V_flat)


@jax.jit
def apply_symmetric_dirichlet(
    csr_values: jnp.ndarray,
    b: jnp.ndarray,
    indices: jnp.ndarray,
    row_per_nnz: jnp.ndarray,
    diag_slots: jnp.ndarray,
    bc_dof_mask: jnp.ndarray,
    bc_increment: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply symmetric Dirichlet BC elimination entirely on device.

    Transforms the system  A * inc = b  into a symmetric reduced system
    by zeroing constrained rows and columns, setting the diagonal to 1, and
    folding the known BC increments into the right-hand side.

    Parameters
    ----------
    csr_values : (nnz_csr,)  — assembled stiffness values.
    b : (n,)                 — right-hand side (negative residual).
    indices : (nnz_csr,) int32 — column index for each CSR slot.
    row_per_nnz : (nnz_csr,) int32 — row index for each CSR slot.
    diag_slots : (n,) int32  — CSR slot of the diagonal A[i,i].
    bc_dof_mask : (n,) bool  — True for constrained DOFs.
    bc_increment : (n,)      — desired Newton increment at each DOF
                               (= u_prescribed - dofs_current for BC DOFs,
                               0 for free DOFs).

    Returns
    -------
    csr_values_bc : (nnz_csr,)  — modified CSR values.
    b_bc : (n,)                 — modified RHS.
    """
    # 1. Fold constrained-column contributions into free-row RHS.
    #    b[row] -= A[row, bc_col] * bc_increment[bc_col]
    col_is_bc = bc_dof_mask[indices]                                 # (nnz_csr,)
    col_contrib = csr_values * bc_increment[indices] * col_is_bc    # zero for free cols
    b = b.at[row_per_nnz].add(-col_contrib)

    # 2. Zero constrained columns in A.
    csr_values = jnp.where(col_is_bc, 0.0, csr_values)

    # 3. Zero constrained rows in A (after the column fold so fold sees original A).
    row_is_bc = bc_dof_mask[row_per_nnz]                            # (nnz_csr,)
    csr_values = jnp.where(row_is_bc, 0.0, csr_values)

    # 4. Restore diagonal = 1 for constrained rows (step 3 zeroed it).
    csr_values = csr_values.at[diag_slots].add(
        bc_dof_mask.astype(csr_values.dtype)
    )

    # 5. Set RHS at constrained DOFs to the prescribed increment.
    b = jnp.where(bc_dof_mask, bc_increment, b)

    return csr_values, b


# ============================================================================
# Build Dirichlet BC metadata (host-side, once per problem)
# ============================================================================

def _build_bc_metadata(problem: Problem, n: int, dtype=jnp.float64):
    """Extract Dirichlet BC DOF mask and prescribed values from a JAX-FEM Problem.

    Compatible with the jax_fem convention where:
    - ``fe.node_inds_list[i]`` is a 1-D numpy array of node indices.
    - ``fe.vec_inds_list[i]`` is a 1-D numpy array of component indices
      (same length as ``node_inds_list[i]``, values in {0, …, vec-1}).
    - ``fe.vals_list[i]`` is a pre-evaluated 1-D JAX/numpy array of
      prescribed values at the selected DOFs.

    Returns
    -------
    bc_dof_mask : jnp.ndarray (n,) bool
    bc_vals_prescribed : jnp.ndarray (n,) float64
        Absolute prescribed displacement at each constrained DOF (0 for free DOFs).
    """
    bc_dof_mask_np = onp.zeros(n, dtype=bool)
    bc_vals_np = onp.zeros(n, dtype=onp.float64)

    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            node_inds = onp.asarray(fe.node_inds_list[i], dtype=onp.int64)
            vec_inds  = onp.asarray(fe.vec_inds_list[i],  dtype=onp.int64)
            vals      = onp.asarray(fe.vals_list[i],       dtype=onp.float64)
            dof_flat  = node_inds * fe.vec + vec_inds + int(problem.offset[ind])
            bc_dof_mask_np[dof_flat] = True
            bc_vals_np[dof_flat] = vals

    return (
        jnp.asarray(bc_dof_mask_np),
        jnp.asarray(bc_vals_np, dtype=dtype),
    )


# ============================================================================
# Newton solver using cuDSS
# ============================================================================

class CuDSSNewtonSolver:
    """Single-step linear solver for JAX-FEM problems using cuDSS.

    All heavy data (CSR pattern, BC metadata) is pre-computed once at
    construction and kept on device.  The solve assembles ``csr_values``
    from ``problem.V_jax`` without touching host memory.

    Only linear problems (``problem.is_linear = True``) are supported on
    the cuDSS path.  For such problems a single Newton step from ``u=0``
    is the exact solution: ``R(u) = Ku - f`` yields ``inc = K⁻¹ f``
    directly, with no residual check required.

    Parameters
    ----------
    problem : DeviceProblem
        The FEM problem instance.  Must be constructed with
        ``gpu_assembly=True`` so that ``self.V_jax`` is populated.
    device_id : int
        cuDSS GPU device index (default 0).
    mtype_id : int
        cuDSS matrix type; derived from ``problem.matrix_symmetry`` by
        :func:`~coil_fem.solvers.build_fwd_pred` (0=general, 1=symmetric,
        3=SPD). Default 1.
    mview_id : int
        cuDSS matrix view (default 0 = full matrix stored).
    """

    def __init__(
        self,
        problem,
        *,
        device_id: int = 0,
        mtype_id: int = 1,
        mview_id: int = 0,
    ):
        self.problem = problem

        n = problem.num_total_dofs_all_vars
        self.n = n

        logger.info("CuDSSNewtonSolver: building CSR pattern …")
        t0 = time.time()
        (
            self.indptr,
            self.indices,
            self.coo_to_csr,
            self.row_per_nnz,
            self.diag_slots,
            self.nnz_csr,
        ) = build_csr_pattern(problem.I, problem.J, n)
        logger.info(
            f"CuDSSNewtonSolver: CSR pattern built in {time.time()-t0:.2f}s  "
            f"n={n}  nnz_csr={self.nnz_csr}"
        )

        logger.info("CuDSSNewtonSolver: building BC metadata …")
        self.bc_dof_mask, self.bc_vals_prescribed = _build_bc_metadata(problem, n)

        logger.info("CuDSSNewtonSolver: instantiating cuDSS solver …")
        CuDSSSolver = _import_cudss_solver()
        # spineax's CuDSSSolver stores csr_offsets / csr_columns as static
        # equinox fields and intentionally puts the (immutable) sparsity-pattern
        # arrays there.  equinox.field(static=True) warns whenever *any* array
        # (JAX or NumPy) is assigned to a static field, so this benign,
        # by-design warning is suppressed here rather than leaked to callers.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A JAX array is being set as static!",
                category=UserWarning,
            )
            self.cudss = CuDSSSolver(
                self.indptr,
                self.indices,
                device_id,
                mtype_id,
                mview_id,
            )

    # ------------------------------------------------------------------
    # Single linear solve (one Newton step)
    # ------------------------------------------------------------------

    def solve_step(
        self,
        b: jnp.ndarray,
        dofs_flat: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve the symmetrically-eliminated system  A_bc * inc = b_bc.

        Parameters
        ----------
        b : (n,)
            Right-hand side **before** BC elimination (negative residual,
            already zero-set at BC rows by ``apply_bc_vec``).
        dofs_flat : (n,)
            Current flat DOF vector (used to compute BC increment).

        Returns
        -------
        inc : (n,)
            Newton increment satisfying ``A_bc * inc = b_bc``.
        inertia : tuple[int, int, int]
            ``(n_positive, n_negative, n_zero)`` eigenvalue counts from
            cuDSS.  Use to verify matrix definiteness (expect ``n_negative=0,
            n_zero=0`` for symmetric positive-definite problems).
        """
        # Assemble CSR values from device Jacobian.
        csr_values = assemble_csr_values(
            self.problem.V_jax, self.coo_to_csr, self.nnz_csr
        )

        # BC increment: desired movement toward prescribed value.
        bc_increment = jnp.where(
            self.bc_dof_mask,
            self.bc_vals_prescribed - dofs_flat,
            0.0,
        )

        # Apply symmetric Dirichlet elimination.
        csr_values_bc, b_bc = apply_symmetric_dirichlet(
            csr_values, b,
            self.indices, self.row_per_nnz, self.diag_slots,
            self.bc_dof_mask, bc_increment,
        )

        # cuDSS solve: A_bc * inc = b_bc
        # inertia = (n_positive, n_negative, n_zero) eigenvalues.
        # For K_cc (symmetric linear-elastic) all n_positive, n_negative=0, n_zero=0.
        inc, inertia = self.cudss(b_bc, csr_values_bc)
        return inc, inertia

    # ------------------------------------------------------------------
    # Single-step linear solve
    # ------------------------------------------------------------------

    def solve(self, params) -> list[jnp.ndarray]:
        """Solve the linear system in a single step.

        Calls ``problem.set_params(params)``, assembles ``K`` and ``f`` via
        ``problem.newton_update``, and solves ``K u = f`` in one cuDSS call
        with no host synchronisation.

        Returns
        -------
        sol_list : list[jnp.ndarray]
            JAX-FEM solution list; ``sol_list[0]`` has shape (n_nodes, vec).
        """
        problem = self.problem
        start = time.time()

        problem.set_params(params)
        dofs = jnp.zeros(self.n)

        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec_bc = apply_bc_vec(res_vec, dofs, problem)

        dofs, inertia = self.solve_step(-res_vec_bc, dofs)
        logger.info(
            f"CuDSSNewtonSolver: linear solve finished in "
            f"{time.time()-start:.2f}s  inertia={inertia}"
        )
        return problem.unflatten_fn_sol_list(dofs)

    # Keep alias so any user code that called newton_loop() still works.
    newton_loop = solve

    # ------------------------------------------------------------------
    # Adjoint linear solve (for custom_vjp backward)
    # ------------------------------------------------------------------

    def adjoint_solve(
        self,
        params,
        sol_list: list[jnp.ndarray],
        g_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve  A^T λ = g  for the adjoint variable λ.

        Since A is symmetric after our elimination,  A^T = A.
        The adjoint BCs are homogeneous (λ[bc_dofs] = 0), so we zero
        g_vec at constrained DOFs before solving.

        Parameters
        ----------
        params : dict
            Forward parameters (used to rebuild A via set_params + newton_update).
        sol_list : list[jnp.ndarray]
            Forward solution.
        g_vec : (n,)
            Flat cotangent vector from the objective's VJP.

        Returns
        -------
        lambda_vec : (n,)
            Adjoint variable.
        """
        problem = self.problem
        problem.set_params(params)
        problem.newton_update(sol_list)

        # Assemble and apply symmetric BC (bc_increment = 0 for adjoint).
        csr_values = assemble_csr_values(
            problem.V_jax, self.coo_to_csr, self.nnz_csr
        )
        bc_increment_zero = jnp.zeros(self.n, dtype=g_vec.dtype)
        # Zero the cotangent at constrained DOFs (homogeneous adjoint BC).
        g_bc = jnp.where(self.bc_dof_mask, 0.0, g_vec)
        csr_values_bc, g_bc2 = apply_symmetric_dirichlet(
            csr_values, g_bc,
            self.indices, self.row_per_nnz, self.diag_slots,
            self.bc_dof_mask, bc_increment_zero,
        )

        lambda_vec, _inertia = self.cudss(g_bc2, csr_values_bc)
        return lambda_vec


# ============================================================================
# VJP-compatible wrapper (drop-in for jax_fem.solver.ad_wrapper)
# ============================================================================

def cudss_ad_wrapper(
    problem,
    *,
    device_id: int = 0,
    mtype_id: int = 1,
    mview_id: int = 0,
):
    """Drop-in replacement for ``jax_fem.solver.ad_wrapper`` using cuDSS.

    Builds a single :class:`CuDSSNewtonSolver` and returns a ``fwd_pred``
    callable decorated with ``jax.custom_vjp``.  The backward pass mirrors
    ``jax_fem.solver.implicit_vjp`` but replaces the PETSc/scipy linear solve
    with cuDSS, which reuses the factorization from the forward pass.

    Only problems with ``is_linear = True`` are accepted; the check is
    enforced in :func:`~coil_fem.solvers.build_fwd_pred` before reaching here.

    Parameters
    ----------
    problem : DeviceProblem (e.g. LinearElasticity3D)
        Must be constructed with ``gpu_assembly=True`` (otherwise ``V_jax``
        is not available and the call will raise ``AttributeError``).
    device_id : int
        GPU device index for cuDSS.
    mtype_id : int
        cuDSS matrix type: 0=general, 1=symmetric, 3=SPD.
        Derived from ``problem.matrix_symmetry`` by :func:`build_fwd_pred`.
    mview_id : int
        cuDSS matrix view: 0=full, 1=lower triangle. Default 0.

    Returns
    -------
    fwd_pred : callable
        ``fwd_pred(params) -> sol_list``, differentiable via ``jax.grad``.
    """
    cudss_solver = CuDSSNewtonSolver(
        problem,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )

    @jax.custom_vjp
    def fwd_pred(params):
        return cudss_solver.solve(params)

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, g_list):
        logger.info("CuDSS backward: solving adjoint problem …")
        params, sol_list = res

        g_vec = jax.flatten_util.ravel_pytree(g_list)[0]
        lambda_vec = cudss_solver.adjoint_solve(params, sol_list, g_vec)

        # VJP of constraint_fn w.r.t. params at (sol_list, params).
        # Mirrors jax_fem.solver.implicit_vjp exactly.
        def constraint_fn(prms):
            problem.set_params(prms)
            res_fn = problem.compute_residual
            res_fn_flat = get_flatten_fn(res_fn, problem)
            res_fn_bc = apply_bc(res_fn_flat, problem)
            dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
            return problem.unflatten_fn_sol_list(res_fn_bc(dofs))

        _primal, f_vjp = jax.vjp(constraint_fn, params)
        (vjp_result,) = f_vjp(problem.unflatten_fn_sol_list(lambda_vec))

        # Negate: adjoint gradient is -λ^T dc/dp.
        vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)
        return (vjp_result,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
