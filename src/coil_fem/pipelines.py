"""Per-coil FEM physics pipelines.

Provides :class:`ElasticPipeline`, which encapsulates all per-coil state
(mesh, problem, forward-prediction callable, and material scalars) previously
spread across lists in :class:`~coil_fem.CoilFEM`.  A stub
:class:`ThermoElasticPipeline` is included for future thermoelastic coupling.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.flatten_util
import jax.numpy as jnp

from .problems import LinearElasticity3D, lame_parameters
from .solvers import build_fwd_pred, needs_gpu_assembly

if TYPE_CHECKING:
    from .meshing import CoilMesh


class ElasticPipeline:
    """All per-coil state for a purely elastic differentiable FEM solve.

    Owns the :class:`~coil_fem.meshing.CoilMesh`, the
    :class:`~coil_fem.problems.LinearElasticity3D` problem instance, and the
    differentiable forward-prediction callable built by
    :func:`~coil_fem.solvers.build_fwd_pred`.

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
    problem_options : dict
        Options forwarded to :func:`~coil_fem.solvers.build_fwd_pred`.
    """

    def __init__(
        self,
        mesh: CoilMesh,
        E: float,
        nu: float,
        itc: float | None,
        gravity_bf: tuple[float, float, float],
        problem_options: dict,
    ):
        self.mesh = mesh
        self.lam, self.mu = lame_parameters(E, nu)
        self.itc = itc

        _use_cudss = needs_gpu_assembly(problem_options)
        thermal_info = (itc,) if itc is not None else (None,)

        self.problem = LinearElasticity3D(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type,
            additional_info=(E, nu, tuple(gravity_bf)) + thermal_info,
            gpu_assembly=_use_cudss,
        )
        mesh.attach_ref_coords(self.problem)

        self.surface_node_indices = self.problem.surface_node_global_indices
        self.problem_options = problem_options

    @functools.cached_property
    def fwd_pred(self):
        """Differentiable forward-prediction callable, built on first access.

        Construction is deferred so that the cuDSS path (which builds the full
        ``CuDSSNewtonSolver``) is skipped entirely on the monolithic coupling
        path where ``fwd_pred`` is never called.
        """
        return build_fwd_pred(self.problem, self.problem_options)

    @property
    def n_surface_quads(self) -> int:
        """Total number of surface quadrature points."""
        return self.problem.n_surface_quads

    def surface_quad_points(self, points: jnp.ndarray) -> jnp.ndarray:
        """Physical positions of all surface quadrature points.

        Differentiable with respect to ``points``.

        Parameters
        ----------
        points : jnp.ndarray, shape ``(n_nodes, 3)``

        Returns
        -------
        jnp.ndarray, shape ``(n_surface_quads, 3)``
        """
        return self.problem.surface_quad_points(points)

    def u_at_surface_quads(self, sol_list: list) -> jnp.ndarray:
        """Interpolate coil displacement to surface quadrature points.

        Maps the nodal displacement field ``sol_list[0]`` to the surface quad
        points using the cached face shape-function values.

        Parameters
        ----------
        sol_list : list[jnp.ndarray]
            Raw ``fwd_pred`` output; ``sol_list[0]`` has shape ``(n_nodes, 3)``.

        Returns
        -------
        jnp.ndarray, shape ``(n_surface_quads, 3)``
        """
        u_surf_nodes = sol_list[0][self.surface_node_indices]  # (n_surf_nodes, 3)
        return self.problem.interp_surface_nodal_to_quads(u_surf_nodes)

    def solve(
        self,
        points: jnp.ndarray,
        body_force: jnp.ndarray,
        support_k: jnp.ndarray,
        *,
        fe_geom: tuple | None = None,
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
        support_k : jnp.ndarray, shape ``(n_surface_quads,)``
            Per-surface-quad Winkler stiffness [N/m³].  Obtain via
            :meth:`surface_quad_points` → ``support.compute_weights`` →
            ``support.stiffness``.
        fe_geom : tuple or None
            Pre-computed ``(shape_grads, JxW, v_grads_JxW, pqp)`` from an
            earlier :func:`~coil_fem.problems.recompute_fe_geometry` call for
            the same ``points``.  When provided, :meth:`~coil_fem.problems.LinearElasticity3D.set_params`
            skips the recompute and uses these arrays directly.

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
            'support_k':   support_k,
        }
        if fe_geom is not None:
            params['_fe_geom'] = fe_geom  # consumed by set_params, not by the solver

        sol_list = self.fwd_pred(params)
        return {
            'sol_list': sol_list,
            'u':        sol_list[0],
            'problem':  self.problem,
        }

    def assemble_coo(self, params: dict) -> tuple:
        """Assemble the stiffness matrix in COO format (cuDSS path only).

        Calls :meth:`~coil_fem.problems.LinearElasticity3D.set_params` then
        triggers a device-side Jacobian assembly via
        :meth:`~coil_fem.problems.DeviceProblem.compute_newton_vars`, populating
        ``problem.V_jax``.  Returns the static ``problem.I`` / ``problem.J``
        index arrays together with the freshly assembled ``problem.V_jax`` and
        the total DOF count.

        Parameters
        ----------
        params : dict
            Same format as accepted by :meth:`solve`.

        Returns
        -------
        tuple
            ``(I, J, V, n_dofs, load)`` where ``I`` and ``J`` are the static
            COO row/column index arrays (host ``numpy``, fixed by the mesh
            topology and used only for building the CSR pattern), ``V`` is the
            freshly assembled COO value array on the JAX device, and ``load``
            is the flat load vector ``-R(0)`` on device.  The load is a
            by-product of the same assembly pass (the residual at zero
            displacement), so callers assembling a merged system need not run
            a separate :meth:`solve_residual`.

        Raises
        ------
        NotImplementedError
            When the pipeline was not built with ``gpu_assembly=True``
            (i.e. ``problem_options={'solver': 'cudss'}`` was not set).
        """
        if not self.problem.gpu_assembly:
            raise NotImplementedError(
                "ElasticPipeline.assemble_coo() requires gpu_assembly=True "
                "(problem_options={'solver': 'cudss'})."
            )
        self.problem.set_params(params)
        zero_sol = [jnp.zeros((self.problem.fes[0].num_total_nodes, 3))]
        res = self.problem.compute_newton_vars(
            zero_sol, self.problem.internal_vars, self.problem.internal_vars_surfaces
        )
        load = -jax.flatten_util.ravel_pytree(res)[0]
        n_dofs = self.problem.num_total_dofs_all_vars
        return self.problem.I, self.problem.J, self.problem.V_jax, n_dofs, load


class ThermoElasticPipeline(ElasticPipeline):
    """Stub pipeline for future thermoelastic coupling (not yet implemented).

    Inherits construction from :class:`ElasticPipeline`; :meth:`solve` raises
    :class:`NotImplementedError` until the heat-conduction + eigenstrain
    coupling is implemented.
    """

    def solve(self, points, body_force, support_k):
        """Not implemented — thermoelastic solve not yet available."""
        raise NotImplementedError(
            "ThermoElasticPipeline.solve() is not yet implemented. "
            "Use ElasticPipeline for purely elastic problems."
        )
