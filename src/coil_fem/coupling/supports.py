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
        curves_jax: list,
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
        curves_jax : list[CurveXYZFourierJAX]
            Differentiable centreline curves for **all** base coils.  The full
            list is required so that beam-network supports can evaluate the
            true beam tangent (which depends on both endpoint curves).
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

    def compute_attach(
        self,
        coil_idx: int,
        surface_pts: jax.Array,
        curves_jax: list,
        dofs,
        state: dict,
    ) -> jax.Array:
        """Per-surface-node target displacement for the shifted Winkler spring.

        In the staggered coupling scheme, the coil-side Winkler spring is shifted
        so that the spring force on coil surface node ``k`` is

        .. math::

            f_k = k_{\\text{lin}} \\cdot w_k \\cdot (u_{\\text{attach},k} - u_k)

        where :math:`u_{\\text{attach},k}` is the attachment displacement returned
        here.  For an uncoupled (grounded) support this is always zero, which
        recovers the standard Winkler spring.

        Parameters
        ----------
        coil_idx : int
            Index of the base coil (0-based).
        surface_pts : jax.Array, shape ``(n_surface_nodes, 3)``
            Current positions of the coil surface nodes.
        curves_jax : list[CurveXYZFourierJAX]
            Differentiable centreline curves for all base coils.
        dofs : dict or None
            Full merged support-dofs dict for the coil set.
        state : dict
            Current support state, including ``'u_s'`` (the support
            displacement vector) when the support is coupled.

        Returns
        -------
        jax.Array, shape ``(n_surface_nodes, 3)``
            Attachment displacement at each surface node.  Default: zeros.
        """
        return jnp.zeros((surface_pts.shape[0], 3), dtype=surface_pts.dtype)

    def coupling_terms(
        self,
        base_curves_dofs,
        support_dofs,
        surface_pts_by_coil: list,
        coil_dof_offsets: list[int],
        support_dof_offset: int,
        surface_node_indices_by_coil: list,
    ) -> dict:
        """COO triplets for the off-diagonal coupling blocks :math:`K_{cs}` and :math:`K_{sc}`.

        In the monolithic assembly the full system is

        .. math::

            \\begin{bmatrix} K_{cc} & K_{cs} \\\\ K_{sc} & K_{ss} \\end{bmatrix}
            \\begin{bmatrix} u_c \\\\ u_s \\end{bmatrix}
            =
            \\begin{bmatrix} f_c \\\\ f_s \\end{bmatrix}

        This method returns the non-zero entries of the off-diagonal blocks as
        three-tuple COO arrays (row index, column index, value).  Row/column
        indices are given in the *global* merged-system DOF numbering.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array]
            Per-base-coil DOF vectors.
        support_dofs : dict
            Merged support-dofs dict for the whole coil set.
        surface_pts_by_coil : list[jax.Array]
            Per-coil surface node positions, shape ``(n_surf_i, 3)`` each.
        coil_dof_offsets : list[int]
            DOF offset of each coil in the merged system.
            ``coil_dof_offsets[i]`` = where coil ``i``'s DOFs start.
        support_dof_offset : int
            DOF offset of the support block in the merged system.
        surface_node_indices_by_coil : list[np.ndarray]
            Per-coil integer arrays ``[n_surf_i]`` mapping surface-node compact
            index ``k`` → global mesh node index (used to locate DOFs in the
            merged system).

        Returns
        -------
        dict with keys ``'I_cs'``, ``'J_cs'``, ``'V_cs'``,
        ``'I_sc'``, ``'J_sc'``, ``'V_sc'``.
            Empty arrays for uncoupled supports.
        """
        empty_i = jnp.zeros(0, dtype=jnp.int32)
        empty_v = jnp.zeros(0, dtype=jnp.float64)
        return {
            'I_cs': empty_i, 'J_cs': empty_i, 'V_cs': empty_v,
            'I_sc': empty_i, 'J_sc': empty_i, 'V_sc': empty_v,
        }


class SupportFixed(Support):
    """Grounded (fixed) support with a configurable Winkler weight function.

    Attachment points are spring-connected to a fixed ground; the effective
    support displacement seen by the coupling driver is always zero.  The
    spatial distribution of the Winkler spring stiffness is controlled by
    the optional ``support_fns`` argument.

    This is the default support used when no coupling to an external
    structural model is needed.  It corresponds to the Winkler / Robin BC
    built into :class:`~coil_fem.problems.LinearElasticity3D`.

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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        curves_jax: list,
        dofs,
    ) -> jax.Array:
        """Per-surface-node Winkler weights for coil ``coil_idx``.

        Parameters
        ----------
        coil_idx : int
            Index of the base coil (0-based).
        surface_pts : jax.Array, shape ``(n_surface_nodes, 3)``
        curves_jax : list[CurveXYZFourierJAX]
            All base-coil centreline curves.  ``curves_jax[coil_idx]`` is
            forwarded to the user ``support_fn``.
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
        curve_i = curves_jax[coil_idx] if curves_jax is not None else None
        return fn(surface_pts, curve_i, dofs)
