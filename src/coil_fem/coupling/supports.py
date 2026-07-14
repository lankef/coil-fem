"""Abstract support interface and the built-in fixed (grounded) support.

Defines :class:`Support`, the ABC that all support structure models must
implement, and :class:`FixedSupport`, the default uncoupled support that
returns zero displacement at all attachment points.
"""

from __future__ import annotations

import abc

import jax
import jax.numpy as jnp


class Support(abc.ABC):
    """Abstract base class for structural support models.

    A ``Support`` models the mechanical boundary between a coil and its
    surrounding structure.  In the uncoupled (grounded) limit the support
    simply returns zero displacement; in coupled problems it maintains its
    own solve state and exposes :meth:`solve` / :meth:`displacement_at` to
    the coupling driver.

    Concrete subclasses
    -------------------
    :class:`FixedSupport`
        Grounded Winkler / Robin BC — attachment points are fixed (zero
        displacement).  Default for uncoupled solves.
    """

    @property
    @abc.abstractmethod
    def is_coupled(self) -> bool:
        """``True`` when the support has its own DOFs that couple to the coil."""

    @abc.abstractmethod
    def solve(self, inputs: dict) -> dict:
        """Advance the support state given coil-side interface data.

        Parameters
        ----------
        inputs : dict
            Driver-supplied data (e.g. attachment-point displacements from
            the coil side).  Uncoupled supports may ignore this.

        Returns
        -------
        dict
            Support state that can be passed back to :meth:`displacement_at`.
            ``FixedSupport`` returns ``{}``.
        """

    @abc.abstractmethod
    def displacement_at(self, state: dict, points: jax.Array) -> jax.Array:
        """Return support-side displacement at the given points.

        Parameters
        ----------
        state : dict
            State dict returned by the most recent :meth:`solve` call.
        points : jax.Array, shape ``(N, 3)``
            Physical coordinates of the query points.

        Returns
        -------
        jax.Array, shape ``(N, 3)``
            Support displacement at each query point.
        """

    def coo(self):
        """Return the support stiffness matrix in COO (coordinate) format. For 
        monolithic mode.

        In the monolithic coupling strategy, the global FEM system is
        assembled as a single block-structured matrix

        .. math::

            \\begin{bmatrix} K_{cc} & K_{cs} \\\\ K_{sc} & K_{ss} \\end{bmatrix}
            \\begin{bmatrix} u_c \\\\ u_s \\end{bmatrix}
            =
            \\begin{bmatrix} f_c \\\\ f_s \\end{bmatrix}

        where subscript *c* denotes coil DOFs and *s* denotes support DOFs.
        The interface coupling blocks :math:`K_{cs}` / :math:`K_{sc}` arise from
        the shared Winkler / contact constraint at the attachment surface.
        ``coo()`` exposes the support's own diagonal block :math:`K_{ss}` (and
        optionally the off-diagonal coupling blocks) so that
        ``solve_monolithic`` can assemble the full system without re-entering
        each sub-problem's internals.

        COO format stores a sparse matrix as three parallel arrays:

        ``I`` : array of int, shape ``(nnz,)``
            Row indices of each non-zero entry, in the *local* DOF numbering
            of the support (0 to ``n_dofs - 1``).
        ``J`` : array of int, shape ``(nnz,)``
            Column indices of each non-zero entry (same numbering as ``I``).
        ``V`` : array of float, shape ``(nnz,)``
            Stiffness values ``K[I[k], J[k]]`` for each entry *k*.  Units are
            N/m (force per displacement).
        ``n_dofs`` : int
            Total number of support DOFs; sets the block size in the global
            matrix.

        The assembled sparse matrix is ``scipy.sparse.coo_matrix((V, (I, J)),
        shape=(n_dofs, n_dofs))`` or its JAX / cuSPARSE equivalent.

        Returns
        -------
        tuple
            ``(I, J, V, n_dofs)`` as described above.

        Raises
        ------
        NotImplementedError
            Raised by default; override in subclasses that support Plan B
            monolithic assembly (e.g. ``BeamNetworkSupport``,
            ``DensityFieldSupport``).  ``FixedSupport`` has no DOFs and never
            contributes a stiffness block, so it intentionally keeps the
            default behaviour.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.coo() is not implemented. "
            "Only supports that participate in monolithic assembly need this."
        )


class FixedSupport(Support):
    """Grounded (fixed) support — attachment points have zero displacement.

    This is the default support used when no coupling to an external
    structural model is needed.  It corresponds to the existing Winkler /
    Robin BC already built into :class:`~coil_fem.problem.LinearElasticity3D`:
    the coil's exterior nodes are spring-connected to a fixed ground, so the
    effective support displacement seen by the coupling driver is zero.

    ``FixedSupport`` stores no state; :meth:`solve` is a no-op and
    :meth:`displacement_at` always returns a zero array.
    """

    @property
    def is_coupled(self) -> bool:
        """``False`` — no DOFs couple to the coil in the fixed-support case."""
        return False

    def solve(self, inputs: dict) -> dict:
        """No-op; returns empty state dict."""
        return {}

    def displacement_at(self, state: dict, points: jax.Array) -> jax.Array:
        """Return zero displacement at every query point.

        Parameters
        ----------
        state : dict
            Ignored.
        points : jax.Array, shape ``(N, 3)``
            Query coordinates.

        Returns
        -------
        jax.Array, shape ``(N, 3)``
            Zero array.
        """
        return jnp.zeros((points.shape[0], 3), dtype=points.dtype)
