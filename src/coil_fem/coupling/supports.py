"""Concrete grounded support and the base class for all support models.

Defines :class:`Support`, the simplest type of support structure. Coils are
fixed to stationary points in space via suspended clamps modelled with
Robin/Winkler boundary conditions.  Which surface nodes are clamped is
controlled by the optional ``fixed_clamp_fns`` callable(s) passed to
:class:`Support`.  Subclasses (e.g. :class:`~coil_fem.coupling.SupportBeams`)
extend the model with inter-coil coupling.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


class Support:
    """Grounded (fixed) support with a configurable Winkler weight function.

    Attachment points are spring-connected to a fixed ground; the effective
    support displacement seen by the coupling driver is always zero.  The
    spatial distribution of the Winkler spring stiffness is controlled by
    the optional ``fixed_clamp_fns`` argument.

    This is the default support used when no coupling to an external
    structural model is needed.  It corresponds to the Winkler / Robin BC
    built into :class:`~coil_fem.problems.LinearElasticity3D`.

    Subclasses
    ----------
    :class:`~coil_fem.coupling.SupportBeams`
        Bisymmetric beam-network model; solves a small linear beam problem
        and exposes coupled DOFs.

    Parameters
    ----------
    fixed_clamp_fns : callable or list[callable] or None
        Function(s) returning per-point weights in ``[0, 1]``::

            fixed_clamp_fn(
                surface_pts : jax.Array,          # (N, 3) — any point cloud
                curve_jax   : CurveXYZFourierJAX,
                dofs_slice  : dict | None,        # per-coil slice, or None
            ) -> jax.Array                        # (N,)

        During solves ``surface_pts`` are the surface **quadrature** points
        ``(n_surface_quads, 3)`` of the coil mesh.  For visualisation the
        function may be called with surface node positions instead.
        ``dofs_slice`` is the per-coil slice of the full dofs dict (each leaf
        indexed at ``coil_idx``), or ``None`` when no optimisation dofs are
        available.  A single callable is broadcast to every coil.  A list
        provides one callable per base coil (heterogeneous support geometry).
        ``None`` (default) returns uniform unit weights for all coils.
    """

    def __init__(
        self,
        fixed_clamp_fns: Callable | list[Callable] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if callable(fixed_clamp_fns):
            self._fixed_clamp_fns: Callable | list[Callable] | None = fixed_clamp_fns
        elif isinstance(fixed_clamp_fns, (list, tuple)):
            self._fixed_clamp_fns = list(fixed_clamp_fns)
        else:
            self._fixed_clamp_fns = None

    @property
    def is_coupled(self) -> bool:
        """``False`` — no DOFs couple to the coil in the grounded-support case."""
        return False

    @property
    def matrix_symmetry(self) -> str:
        """``'symmetric'`` — unconditional after ``k_attachment`` unification.

        All support-block contributions (translational springs, moment-arm
        cross-terms) are symmetric by construction when a single
        ``k_attachment`` modulus is used.  :class:`SupportBeams` inherits
        this without an override.
        """
        return 'symmetric'

    @property
    def k_attachment(self) -> float:
        """Distributed attachment (Winkler) modulus [N/m³].

        Must be implemented by coupled :class:`Support` subclasses and must
        equal ``problem_options['winkler_k']`` so the coil-side and beam-side
        spring sums use the same modulus.

        Raises
        ------
        NotImplementedError
            When the subclass has not declared its attachment modulus.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement the `k_attachment` property "
            "and set it equal to problem_options['winkler_k']."
        )

    def solve(self, inputs: dict) -> dict:
        """No-op; returns empty state dict."""
        return {}

    def geometry(self, curves_jax: list, support_dofs: dict) -> dict | None:
        """Precomputed geometry bundle for this support; ``None`` for uncoupled supports.

        Subclasses that benefit from single-pass geometry computation (e.g.
        :class:`~coil_fem.coupling.SupportBeams`) override this method to
        return a ``dict`` of traced arrays that can be threaded through
        :meth:`compute_weights`, :meth:`compute_attach`, :meth:`coo`, and
        :meth:`coupling_values` via their ``geom`` keyword argument, avoiding
        redundant recomputation.

        Parameters
        ----------
        curves_jax : list[CurveXYZFourierJAX]
            Differentiable centreline curves for all base coils.
        support_dofs : dict
            Optimisable support parameters.

        Returns
        -------
        dict or None
        """
        return None

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
        surface_pts : jax.Array, shape ``(N, 3)``
            Query points.  During solves these are the surface quadrature
            points ``(n_surface_quads, 3)``; for visualisation they may be
            surface node positions.  The callable is shape-agnostic.
        curves_jax : list[CurveXYZFourierJAX]
            Differentiable centreline curves for **all** base coils.  The full
            list is required so that beam-network supports can evaluate the
            true beam tangent (which depends on both endpoint curves).
        dofs : dict or None
            Optimisable support parameters for the full coil set (as
            returned by :attr:`~coil_fem.simsopt.CoilSupport.support_dofs`).
            Before calling ``fixed_clamp_fns``, each leaf is indexed at
            ``coil_idx`` via :func:`jax.tree_util.tree_map`, so the callable
            receives a per-coil slice.  ``None`` is passed through as-is.

        Returns
        -------
        jax.Array, shape ``(N,)``
            Winkler weight in ``[0, 1]`` for each query point.
        """
        if self._fixed_clamp_fns is None:
            return jnp.ones(surface_pts.shape[0])
        if isinstance(self._fixed_clamp_fns, list):
            fn = self._fixed_clamp_fns[coil_idx]
        else:
            fn = self._fixed_clamp_fns
        curve_i = curves_jax[coil_idx] if curves_jax is not None else None
        dofs_slice = (
            jax.tree_util.tree_map(lambda x: x[coil_idx], dofs)
            if dofs is not None else None
        )
        return fn(surface_pts, curve_i, dofs_slice)

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
            assembly (e.g. ``SupportBeams``, ``DensityFieldSupport``).
            The grounded :class:`Support` has no DOFs and never contributes a
            stiffness block, so it intentionally keeps the default behaviour.
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
        """Per-query-point target displacement for the shifted Winkler spring.

        In the staggered coupling scheme, the coil-side Winkler spring is shifted
        so that the spring force at each surface quad point ``q`` is

        .. math::

            f_q = k_{\\text{lin}} \\cdot w_q \\cdot (u_{\\text{attach},q} - u_q)

        where :math:`u_{\\text{attach},q}` is the attachment displacement returned
        here.  For an uncoupled (grounded) support this is always zero, which
        recovers the standard Winkler spring.

        Parameters
        ----------
        coil_idx : int
            Index of the base coil (0-based).
        surface_pts : jax.Array, shape ``(N, 3)``
            Query points; during solves these are surface quadrature points.
        curves_jax : list[CurveXYZFourierJAX]
            Differentiable centreline curves for all base coils.
        dofs : dict or None
            Full merged support-dofs dict for the coil set.
        state : dict
            Current support state, including ``'u_s'`` (the support
            displacement vector) when the support is coupled.

        Returns
        -------
        jax.Array, shape ``(N, 3)``
            Attachment displacement at each query point.  Default: zeros.
        """
        return jnp.zeros((surface_pts.shape[0], 3), dtype=surface_pts.dtype)

    def coupling_pattern(
        self,
        coil_dof_offsets: list[int],
        support_dof_offset: int,
        surface_node_indices_by_coil: list,
    ) -> tuple:
        """Static numpy I/J index arrays for K_cs and K_sc coupling blocks.

        Parameters
        ----------
        coil_dof_offsets : list[int]
        support_dof_offset : int
        surface_node_indices_by_coil : list[np.ndarray]

        Returns
        -------
        tuple of four np.ndarray (I_cs, J_cs, I_sc, J_sc), all empty for the
        uncoupled base support.
        """
        import numpy as onp
        empty = onp.zeros(0, dtype=onp.int32)
        return empty, empty, empty, empty

    def coupling_values(
        self,
        curves_jax: list,
        support_dofs: dict,
        surface_pts_by_coil: list,
        surf_interp_by_coil=None,
        *,
        jxw_by_coil: list,
        geom: dict | None = None,
    ) -> tuple:
        """Traced V arrays for K_cs and K_sc coupling blocks.

        Parameters
        ----------
        curves_jax : list[CurveXYZFourierJAX]
        support_dofs : dict
        surface_pts_by_coil : list[jax.Array]
            Surface query points.  Subclasses may receive surface quadrature
            points ``(n_sq_i, 3)`` when ``surf_interp_by_coil`` is provided.
        surf_interp_by_coil : list or None
            Per-coil interpolation maps for folding quad-point weights back to
            per-node DOF quantities (ignored for the uncoupled base support).
        jxw_by_coil : list[jax.Array]
            Per-coil JxW arrays ``(num_sel, n_fq)`` from
            :meth:`~coil_fem.problems.LinearElasticity3D.surface_jxw`.
            Ignored for the uncoupled base support.
        geom : dict or None
            Pre-computed beam geometry (ignored for the uncoupled base support).

        Returns
        -------
        tuple (V_cs, V_sc) of empty jax.Arrays for the uncoupled base support.
        """
        return jnp.zeros(0), jnp.zeros(0)

