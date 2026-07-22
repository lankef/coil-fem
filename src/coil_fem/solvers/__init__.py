"""FEM solver helpers for selecting and constructing forward-prediction callables.

Provides :func:`needs_gpu_assembly` and :func:`build_fwd_pred` to centralise the
CPU/GPU solver selection that was previously inlined in :class:`~coil_fem.CoilFEM`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..problems.linear_elasticity import LinearElasticity3D


def needs_gpu_assembly(problem_options: dict) -> bool:
    """Return ``True`` when the configured solver requires on-device assembly.

    Parameters
    ----------
    problem_options : dict
        Problem options dict, typically containing ``'solver'``.

    Returns
    -------
    bool
        ``True`` iff ``problem_options['solver'] == 'cudss'``.
    """
    return problem_options.get('solver', 'umfpack') == 'cudss'


def build_fwd_pred(problem: LinearElasticity3D, problem_options: dict):
    """Build a differentiable forward-prediction callable for ``problem``.

    Selects either the standard CPU ``ad_wrapper`` or the GPU
    ``cudss_ad_wrapper`` based on ``problem_options['solver']``.  The
    ``cudss_ad_wrapper`` import is lazy so that CPU-only installs remain
    unaffected.

    Parameters
    ----------
    problem : LinearElasticity3D
        Fully constructed JAX-FEM problem (``custom_init`` already called).
    problem_options : dict
        Options dict.  Recognised keys:

        ``'solver'`` : str, default ``'umfpack'``
        ``'adjoint_solver'`` : str, default ``'umfpack'``
        ``'cudss_device_id'`` : int, default ``0``
        ``'cudss_mtype_id'``  : int, default ``1``
        ``'cudss_tol'``       : float, default ``1e-6``
        ``'cudss_rel_tol'``   : float, default ``1e-8``
        ``'cudss_max_iter'``  : int, default ``50``

    Returns
    -------
    callable
        ``fwd_pred(params: dict) -> list[jnp.ndarray]`` — a differentiable
        function that calls ``problem.set_params(params)`` and solves the
        linear system, returning ``sol_list`` in JAX-FEM's multi-physics
        convention.
    """
    from jax_fem.solver import ad_wrapper

    _use_cudss = needs_gpu_assembly(problem_options)

    if _use_cudss:
        from .cudss import cudss_ad_wrapper
        return cudss_ad_wrapper(
            problem,
            device_id=int(problem_options.get('cudss_device_id', 0)),
            mtype_id=int(problem_options.get('cudss_mtype_id', 1)),
            tol=float(problem_options.get('cudss_tol', 1e-6)),
            rel_tol=float(problem_options.get('cudss_rel_tol', 1e-8)),
            max_iter=int(problem_options.get('cudss_max_iter', 50)),
        )

    solver_name     = problem_options.get('solver', 'umfpack')
    adj_solver_name = problem_options.get('adjoint_solver', 'umfpack')
    solver_opts     = {f"{solver_name}_solver": {}}
    adj_solver_opts = {f"{adj_solver_name}_solver": {}}
    return ad_wrapper(
        problem,
        solver_options=solver_opts,
        adjoint_solver_options=adj_solver_opts,
    )
