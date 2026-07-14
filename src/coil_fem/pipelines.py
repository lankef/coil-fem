"""Per-coil FEM physics pipelines.

Provides :class:`ElasticPipeline`, which encapsulates all per-coil state
(mesh, problem, forward-prediction callable, and material scalars) previously
spread across lists in :class:`~coil_fem.CoilFEM`.  A stub
:class:`ThermoElasticPipeline` is included for future thermoelastic coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from .problem import LinearElasticity3D, lame_parameters, itc_strain
from .solver import build_fwd_pred, needs_gpu_assembly

if TYPE_CHECKING:
    from .meshing import CoilMesh


class ElasticPipeline:
    """All per-coil state for a purely elastic differentiable FEM solve.

    Owns the :class:`~coil_fem.meshing.CoilMesh`, the
    :class:`~coil_fem.problem.LinearElasticity3D` problem instance, and the
    differentiable forward-prediction callable built by
    :func:`~coil_fem.solver.build_fwd_pred`.

    Parameters
    ----------
    mesh : CoilMesh
        Coil cross-section mesh.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    itc : float or None
        Integral thermal contraction ``ΔL/L`` (positive, dimensionless).
        ``None`` for isothermal.
    gravity_bf : tuple[float, float, float]
        Constant gravity body-force component ``ρ g_vec`` [N/m³].
    winkler_k : float
        Base Winkler spring stiffness [N/m³].
    problem_options : dict
        Options forwarded to :func:`~coil_fem.solver.build_fwd_pred`.
    """

    def __init__(
        self,
        mesh: CoilMesh,
        E: float,
        nu: float,
        itc: float | None,
        gravity_bf: tuple[float, float, float],
        winkler_k: float,
        problem_options: dict,
    ):
        self.mesh = mesh
        self.lam, self.mu = lame_parameters(E, nu)
        self.itc = itc

        _use_cudss = needs_gpu_assembly(problem_options)
        thermal_info = (itc,) if itc is not None else (None,)

        self.problem = LinearElasticity3D(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type,
            additional_info=(E, nu, tuple(gravity_bf), winkler_k) + thermal_info,
            gpu_assembly=_use_cudss,
        )
        mesh.attach_ref_coords(self.problem)

        self.fwd_pred = build_fwd_pred(self.problem, problem_options)
        self.surface_node_indices = self.problem.surface_node_global_indices

    def solve(
        self,
        points: jnp.ndarray,
        body_force: jnp.ndarray,
        support_weights: jnp.ndarray | None = None,
    ) -> dict:
        """Run one differentiable forward FEM solve.

        Builds uniform per-quad material arrays from the scalar ``lam``/``mu``
        stored at construction and calls ``fwd_pred``.

        Parameters
        ----------
        points : jnp.ndarray, shape ``(n_nodes, 3)``
            Current mesh node positions.
        body_force : jnp.ndarray, shape ``(n_cells, n_quads, 3)``
            Body force at every quadrature point.
        support_weights : jnp.ndarray or None, shape ``(n_surface_nodes,)``
            Per-surface-node Winkler weights in ``[0, 1]``.

        Returns
        -------
        dict
            ``'sol_list'``  — raw ``ad_wrapper`` output, list of length 1.
            ``'u'``         — displacement field, shape ``(n_nodes, 3)``.
            ``'problem'``   — reference to ``self.problem`` for post-processing.
        """
        params: dict = {
            'points':      points,
            'body_force':  body_force,
        }
        if support_weights is not None:
            params['support_weights'] = support_weights

        sol_list = self.fwd_pred(params)
        return {
            'sol_list': sol_list,
            'u':        sol_list[0],
            'problem':  self.problem,
        }

    def attachment_displacement(self, sol_list: list) -> jnp.ndarray:
        """Extract displacement at coil surface (attachment) nodes.

        Parameters
        ----------
        sol_list : list[jnp.ndarray]
            Raw ``fwd_pred`` output.

        Returns
        -------
        jnp.ndarray, shape ``(n_surface_nodes, 3)``
        """
        return sol_list[0][self.surface_node_indices]

    def coo(self):
        """Return assembled stiffness matrix in COO format (Plan B preparation).

        Only available when the problem was built with ``gpu_assembly=True``
        (the cuDSS solver path).

        Returns
        -------
        tuple
            ``(I, J, V, n_dofs)`` — row indices, column indices, values, and
            total number of DOFs.

        Raises
        ------
        NotImplementedError
            When the problem was built without on-device assembly
            (``gpu_assembly=False``, i.e. the CPU / ``ad_wrapper`` path).
        """
        if not hasattr(self.problem, 'I_jax'):
            raise NotImplementedError(
                "ElasticPipeline.coo() requires gpu_assembly=True "
                "(problem_options={'solver': 'cudss'})."
            )
        n_dofs = self.problem.num_total_dofs_all_vars
        return self.problem.I_jax, self.problem.J_jax, self.problem.V_jax, n_dofs


class ThermoElasticPipeline(ElasticPipeline):
    """Stub pipeline for future thermoelastic coupling (not yet implemented).

    Inherits construction from :class:`ElasticPipeline`; :meth:`solve` raises
    :class:`NotImplementedError` until the heat-conduction + eigenstrain
    coupling is implemented.
    """

    def solve(self, points, body_force, support_weights=None):
        """Not implemented — thermoelastic solve not yet available."""
        raise NotImplementedError(
            "ThermoElasticPipeline.solve() is not yet implemented. "
            "Use ElasticPipeline for purely elastic problems."
        )
