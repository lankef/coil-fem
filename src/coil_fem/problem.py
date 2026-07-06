"""Customized JAX-FEM ``Problem`` for the cuDSS GPU backend.

:class:`DeviceProblem` is a drop-in subclass of :class:`jax_fem.problem.Problem`
introduced **solely for compatibility with the cuDSS solver backend**
(``problem_options={'solver': 'cudss'}``).  It exists so that the on-device
Jacobian-assembly behaviour can be toggled with a single constructor flag
instead of synthesizing temporary subclasses / mixins at runtime.

Why this class exists
---------------------
JAX-FEM's default :meth:`Problem.compute_newton_vars` copies the per-cell
Jacobian to host (``self.V = onp.array(cells_jac_flat.reshape(-1))``) before
building a scipy/PETSc CSR matrix.  The cuDSS backend instead consumes the
Jacobian values directly on the JAX device, so this host round-trip is pure
overhead.  When ``gpu_assembly=True`` the override keeps the flat COO
Jacobian values in ``self.V_jax`` (a JAX device array) and never populates the
host ``self.V``.

Design notes
------------
* The override only uses **generic** ``Problem`` machinery
  (``cells_list``, ``split_and_compute_cell``, ``compute_face``,
  ``compute_residual_vars_helper``); it is not specific to any particular
  physics, so it lives at the ``Problem`` layer and can be reused by any
  FEM problem that wants the cuDSS backend.
* It deliberately has **no spineax / cuDSS import**.  The device-assembly path
  needs only ``jax`` + ``jax_fem``, so importing ``DeviceProblem`` is always
  safe even on CPU-only machines.  The heavy spineax/cuDSS dependency is pulled
  in lazily, only when the cuDSS solver is actually constructed
  (see :mod:`coil_fem.backend.cudss`).
* When ``gpu_assembly=False`` (the default) the class is byte-for-byte
  equivalent to the stock ``Problem`` — ``compute_newton_vars`` simply defers
  to ``super()``.
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

    This is a thin compatibility shim for the cuDSS backend; see the module
    docstring for the rationale.

    Parameters
    ----------
    gpu_assembly : bool, default False
        When ``True``, :meth:`compute_newton_vars` keeps the flat COO Jacobian
        values on the JAX device in ``self.V_jax`` and skips the host
        ``self.V`` copy.  When ``False``, behaviour is identical to the stock
        :class:`jax_fem.problem.Problem`.

        After each call to :meth:`Problem.newton_update`, ``self.V_jax`` holds
        the flat COO Jacobian values; ``self.V`` (host numpy) is **not**
        updated.

    Notes
    -----
    ``gpu_assembly`` is a dataclass field placed after all of ``Problem``'s
    fields (which all carry defaults), so it can be selected at construction::

        problem = LinearElasticity3D(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type,
            additional_info=(...),
            gpu_assembly=True,   # enable the cuDSS-compatible path
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
