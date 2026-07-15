"""Abstract support interface and the built-in fixed (grounded) support.

Defines :class:`Support`, the ABC that all support structure models must
implement, and :class:`SupportFixed`, the default uncoupled support whose
attachment points are held at zero displacement by a Winkler spring field
whose spatial distribution is governed by a user-supplied weight function.
"""

from __future__ import annotations

import abc
from typing import Callable

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
    :class:`SupportFixed`
        Grounded Winkler / Robin BC — attachment points are fixed (zero
        displacement).  Default for uncoupled solves.
    ``BeamNetworkSupport`` *(future)*
        Beam-network model; solves a small linear beam problem.
    ``DensityFieldSupport`` *(future)*
        Density-parameterised FEM support.
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
            ``SupportFixed`` returns ``{}``.
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

    def compute_weights(
        self,
        coil_idx: int,
        surface_pts: jax.Array,
        curve_jax,
        dofs,
    ) -> jax.Array:
        """Per-surface-node Winkler weights for coil ``coil_idx``.

        Weights are values in ``[0, 1]`` used to scale the Winkler spring
        stiffness at each surface node.  The default implementation returns
        a uniform weight of one (fully supported everywhere).  Override in
        subclasses that parameterise the spatial distribution of support.

        Parameters
        ----------
        coil_idx : int
            Index of the base coil (0-based).
        surface_pts : jax.Array, shape ``(n_surface_nodes, 3)``
            Current positions of the coil surface nodes.
        curve_jax : CurveXYZFourierJAX
            Differentiable representation of the coil centreline at the
            current DOFs.
        dofs : dict or None
            Optimisable support parameters for the full coil set (as
            returned by :attr:`~coil_fem.simsopt.CoilSupport.support_dofs`).
            Subclasses are responsible for slicing out the per-coil entries.
            ``None`` is equivalent to an empty dict.

        Returns
        -------
        jax.Array, shape ``(n_surface_nodes,)``
            Winkler weight in ``[0, 1]`` for each surface node.
        """
        return jnp.ones(surface_pts.shape[0])

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
            Raised by default; override in subclasses that support monolithic
            assembly (e.g. ``BeamNetworkSupport``, ``DensityFieldSupport``).
            ``SupportFixed`` has no DOFs and never contributes a stiffness
            block, so it intentionally keeps the default behaviour.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.coo() is not implemented. "
            "Only supports that participate in monolithic assembly need this."
        )


class SupportFixed(Support):
    """Grounded (fixed) support with a configurable Winkler weight function.

    Attachment points are spring-connected to a fixed ground; the effective
    support displacement seen by the coupling driver is always zero.  The
    spatial distribution of the Winkler spring stiffness is controlled by
    the optional ``support_fns`` argument.

    This is the default support used when no coupling to an external
    structural model is needed.  It corresponds to the Winkler / Robin BC
    built into :class:`~coil_fem.problem.LinearElasticity3D`.

    Parameters
    ----------
    support_fns : callable or list[callable] or None
        Function(s) returning per-surface-node weights in ``[0, 1]``::

            support_fn(
                surface_pts: jax.Array,   # (n_surface_nodes, 3)
                curve_jax: CurveXYZFourierJAX,
                dofs: dict | None,
            ) -> jax.Array                # (n_surface_nodes,)

        A single callable is broadcast to every coil.  A list provides one
        callable per base coil (heterogeneous support geometry).  ``None``
        (default) returns uniform unit weights for all coils.
    """

    def __init__(
        self,
        support_fns: Callable | list[Callable] | None = None,
    ):
        if callable(support_fns):
            self._support_fns: Callable | list[Callable] | None = support_fns
        elif isinstance(support_fns, (list, tuple)):
            self._support_fns = list(support_fns)
        else:
            self._support_fns = None

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

    def compute_weights(
        self,
        coil_idx: int,
        surface_pts: jax.Array,
        curve_jax,
        dofs,
    ) -> jax.Array:
        """Per-surface-node Winkler weights for coil ``coil_idx``.

        Parameters
        ----------
        coil_idx : int
            Index of the base coil (0-based).
        surface_pts : jax.Array, shape ``(n_surface_nodes, 3)``
        curve_jax : CurveXYZFourierJAX
        dofs : dict or None
            Full merged support-dofs dict for the coil set; subclasses
            slice out the per-coil portion as needed.

        Returns
        -------
        jax.Array, shape ``(n_surface_nodes,)``
        """
        if self._support_fns is None:
            return jnp.ones(surface_pts.shape[0])
        if isinstance(self._support_fns, list):
            fn = self._support_fns[coil_idx]
        else:
            fn = self._support_fns
        return fn(surface_pts, curve_jax, dofs)
