"""FEM solver helpers for selecting and constructing forward-prediction callables.

Provides :func:`needs_gpu_assembly` and :func:`build_fwd_pred` to centralise the
CPU/GPU solver selection that was previously inlined in :class:`~coil_fem.CoilFEM`.
"""

from __future__ import annotations

import warnings
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

    The cuDSS matrix type (``mtype_id``) is derived from
    ``problem.matrix_symmetry`` (e.g. ``'symmetric'`` → 1).  Pass
    ``problem_options['cudss_mtype_id']`` only to override the derived value;
    a warning is emitted when the override disagrees with the derived claim.

    Parameters
    ----------
    problem : LinearElasticity3D
        Fully constructed JAX-FEM problem (``custom_init`` already called).
    problem_options : dict
        Options dict.  Recognised keys:

        ``'solver'`` : str, default ``'umfpack'``
        ``'adjoint_solver'`` : str, default ``'umfpack'``
        ``'cudss_device_id'`` : int, default ``0``
        ``'cudss_mtype_id'``  : int, optional override — derived from
            ``problem.matrix_symmetry`` by default; emits ``UserWarning``
            when it disagrees with the derived value.

    Returns
    -------
    callable
        ``fwd_pred(params: dict) -> list[jnp.ndarray]`` — a differentiable
        function that calls ``problem.set_params(params)`` and solves the
        linear system, returning ``sol_list`` in JAX-FEM's multi-physics
        convention.

    Raises
    ------
    NotImplementedError
        When ``solver='cudss'`` and ``problem.is_linear`` is ``False``.
        Use jax-fem's CPU ``ad_wrapper`` for non-linear problems.
    """
    from jax_fem.solver import ad_wrapper

    _use_cudss = needs_gpu_assembly(problem_options)

    if _use_cudss:
        from .cudss import cudss_ad_wrapper, _MTYPE_ID

        if not getattr(problem, 'is_linear', False):
            raise NotImplementedError(
                f"{type(problem).__name__} does not declare is_linear=True. "
                "The cuDSS path requires a linear problem (single-step solve). "
                "Use solver='umfpack' or another CPU backend for non-linear problems."
            )

        # Derive mtype_id from problem's declared symmetry.
        derived_sym = getattr(problem, 'matrix_symmetry', 'symmetric')
        mtype_id = _MTYPE_ID[derived_sym]
        if 'cudss_mtype_id' in problem_options:
            override = int(problem_options['cudss_mtype_id'])
            if override != mtype_id:
                warnings.warn(
                    f"cudss_mtype_id={override} overrides derived value "
                    f"{mtype_id} (from matrix_symmetry={derived_sym!r}). "
                    "Verify this is intentional.",
                    stacklevel=2,
                )
            mtype_id = override

        return cudss_ad_wrapper(
            problem,
            device_id=int(problem_options.get('cudss_device_id', 0)),
            mtype_id=mtype_id,
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
