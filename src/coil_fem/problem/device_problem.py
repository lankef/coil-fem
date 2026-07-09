"""JAX-FEM ``Problem`` variant whose Jacobian can stay on the JAX device.

:class:`DeviceProblem` is a thin drop-in subclass of
:class:`jax_fem.problem.Problem`.  Setting ``gpu_assembly=True`` keeps the
flat COO Jacobian values in ``self.V_jax`` (a JAX device array) and skips
the host copy normally done by ``compute_newton_vars``, making it compatible
with the cuDSS GPU solver backend.  The default ``gpu_assembly=False``
behaviour is byte-for-byte identical to the stock ``Problem``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.flatten_util
import jax.numpy as jnp

from jax_fem import logger
from jax_fem.problem import Problem


@dataclass
class DeviceProblem(Problem):
    """``Problem`` variant whose Jacobian assembly can stay on the JAX device.

    Parameters
    ----------
    gpu_assembly : bool, default False
        When ``True``, keeps the flat COO Jacobian in ``self.V_jax`` and
        skips the host copy in :meth:`compute_newton_vars`.  Required for the
        cuDSS backend; harmless otherwise.

    Examples
    --------
    ::

        problem = LinearElasticity3D(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type,
            additional_info=(...),
            gpu_assembly=True,
        )
    """

    gpu_assembly: bool = False

    def compute_newton_vars(self, sol_list, internal_vars, internal_vars_surfaces):
        """Jacobian + residual assembly.

        Defers to the stock host-assembly path unless ``gpu_assembly`` is
        set, in which case the flat COO Jacobian values are kept on the JAX
        device in ``self.V_jax`` (no host round-trip).
        """
        if not self.gpu_assembly:
            return super().compute_newton_vars(
                sol_list, internal_vars, internal_vars_surfaces
            )

        logger.debug("DeviceProblem: computing Jacobian on device")
        cells_sol_list = [
            sol[cells]
            for cells, sol in zip(self.cells_list, sol_list)
        ]
        cells_sol_flat = jax.vmap(
            lambda *x: jax.flatten_util.ravel_pytree(x)[0]
        )(*cells_sol_list)

        # Use jnp so vstack keeps tensors on device.
        weak_form_flat, cells_jac_flat = self.split_and_compute_cell(
            cells_sol_flat, jnp, True, internal_vars
        )
        weak_form_face_flat, cells_jac_face_flat = self.compute_face(
            cells_sol_flat, jnp, True, internal_vars_surfaces
        )

        # Concatenate all flat Jacobian contributions on device.
        V_parts = [cells_jac_flat.reshape(-1)] + [
            j.reshape(-1) for j in cells_jac_face_flat
        ]
        self.V_jax = jnp.concatenate(V_parts)   # stays on GPU

        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)
