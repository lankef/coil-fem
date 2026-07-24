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
        Function(s) returning per-surface-node weights in ``[0, 1]``::

            fixed_clamp_fn(
                surface_pts : jax.Array,          # (n_surface_nodes, 3)
                curve_jax   : CurveXYZFourierJAX,
                dofs_slice  : dict | None,        # per-coil slice, or None
            ) -> jax.Array                        # (n_surface_nodes,)

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
            Current positions of the coil surface nodes.
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
        jax.Array, shape ``(n_surface_nodes,)``
            Winkler weight in ``[0, 1]`` for each surface node.
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

    def plot_support(
        self,
        fem,
        *,
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: dict | None = None,
        ax=None,
        s: float = 0.1,
        cmap: str = "viridis",
        color="C0",
        simple_mode: bool = False,
        **kwargs,
    ):
        """Scatter-plot the mesh nodes of every base coil coloured by Winkler weight.

        Parameters
        ----------
        fem : coil_fem.CoilFEM
            The FEM container owning this support.  Supplies the per-coil
            meshes, pipelines, base curves, and the Winkler-weight helper.
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``fem.base_curves_jax``.
        base_support_dofs : dict or None
            Per-coil support parameters for the support functions.  ``None``
            (default) uses the support parameters supplied at construction.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None
            Existing 3-D axes to draw on.  ``None`` (default) creates a new
            figure and 3-D axes.
        s : float
            Marker size for the scatter (default ``0.1``).
        cmap : str
            Matplotlib colormap name for the support weights (default
            ``"viridis"``).  Ignored when ``simple_mode`` is ``True``.
        color : color-like
            Single marker colour used only when ``simple_mode`` is ``True``
            (default ``"C0"``).
        simple_mode : bool
            When ``True``, disable the colormap and colorbar: every point is
            drawn in a single ``color`` and the support weight (guaranteed in
            ``[0, 1]``) is used as each point's **alpha**, so fully supported
            nodes are opaque and free nodes are invisible.
        **kwargs
            Extra keyword arguments forwarded to :meth:`ax.scatter`
            (e.g. ``marker``, ``facecolors``, ``edgecolors``).

        Returns
        -------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            The 3-D axes used for the plot.  The parent figure is available as
            ``ax.get_figure()``.
        """
        import numpy as onp
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb

        from ..geo import CurveXYZFourierJAX as _CurveJAX

        n_base = len(fem.base_curves_jax)
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in fem.base_curves_jax]

        curves_jax = [
            _CurveJAX(base.quadpoints, d, base.order)
            for base, d in zip(fem.base_curves_jax, base_curves_dofs)
        ]

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = ax.get_figure()

        sc = None
        for i in range(n_base):
            pts_i  = fem.meshes[i].mesh_points_from_dofs(base_curves_dofs[i])
            pts_np = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(fem.pipelines[i].surface_node_indices, dtype=onp.int32)
            weights_surf = onp.asarray(
                fem._compute_support_weights(i, pts_i, curves_jax, base_support_dofs),
                dtype=onp.float64,
            )
            weight_full = onp.zeros(n_nodes, dtype=onp.float64)
            weight_full[surf_idx] = weights_surf

            if simple_mode:
                # No colormap/colorbar: encode weight as per-point alpha.
                # Build an explicit (n_nodes, 4) RGBA array so per-point
                # transparency survives even for hollow markers.
                rgba = onp.empty((n_nodes, 4), dtype=onp.float64)
                rgba[:, :3] = to_rgb(color)
                rgba[:, 3] = onp.clip(weight_full, 0.0, 1.0)
                # Route the per-point colour to the edges for hollow markers
                # (``facecolors="none"``) and to the faces otherwise, avoiding
                # the ``c``-vs-``facecolors`` precedence conflict.
                scatter_kw = dict(kwargs)
                if str(scatter_kw.get("facecolors")) == "none":
                    scatter_kw.setdefault("edgecolors", rgba)
                else:
                    scatter_kw.setdefault("facecolors", rgba)
                ax.scatter(
                    pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                    s=s, **scatter_kw,
                )
            else:
                sc = ax.scatter(
                    pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                    s=s, c=weight_full, cmap=cmap, vmin=0.0, vmax=1.0,
                    **kwargs,
                )
        if sc is not None and not simple_mode:
            fig.colorbar(sc, ax=ax, label="support weight")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        return ax

    def save_support_vtu(
        self,
        fem,
        out_dir: str = ".",
        *,
        prefix: str = "coil",
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: dict | None = None,
    ) -> list[str]:
        """Export Winkler support weights and full mesh as VTU files.

        For each base coil ``i``, writes:

        * ``{out_dir}/{prefix}{i:02d}_support.vtu`` — full tetrahedral mesh with:

          - point field ``support_weights`` in ``[0, 1]``; ``1`` = fully
            supported, ``0`` = free.
          - point field ``spring_k_Npm3`` — effective Winkler spring stiffness
            ``winkler_k × support_weight`` in N/m³.

        Open in ParaView; use *Filters → Threshold* on ``support_weights`` or
        ``spring_k_Npm3`` to isolate the clamped region.

        Parameters
        ----------
        fem : coil_fem.CoilFEM
            The FEM container owning this support.  Supplies the per-coil
            meshes, pipelines, base curves, Winkler stiffness, and the VTU
            writer helper.
        out_dir : str
            Output directory.  Created if it does not exist.
        prefix : str
            File-name prefix (default ``"coil"``).
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``fem.base_curves_jax``.
        base_support_dofs : dict or None
            Per-coil support parameters for the support functions.

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        import os
        import numpy as onp

        from ..geo import CurveXYZFourierJAX as _CurveJAX

        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in fem.base_curves_jax]

        curves_jax = [
            _CurveJAX(base.quadpoints, d, base.order)
            for base, d in zip(fem.base_curves_jax, base_curves_dofs)
        ]

        winkler_k = float(fem.problem_options['winkler_k'])

        os.makedirs(out_dir, exist_ok=True)
        written: list[str] = []

        for i, coil_mesh in enumerate(fem.meshes):
            pts_i  = fem.meshes[i].mesh_points_from_dofs(base_curves_dofs[i])
            pts_np = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(fem.pipelines[i].surface_node_indices, dtype=onp.int32)
            weights_surf = onp.asarray(
                fem._compute_support_weights(i, pts_i, curves_jax, base_support_dofs),
                dtype=onp.float64,
            )
            weight_full = onp.zeros(n_nodes, dtype=onp.float64)
            weight_full[surf_idx] = weights_surf

            mesh_path = os.path.join(out_dir, f"{prefix}{i:02d}_support.vtu")
            fem._write_coil_vtu(
                mesh_path, coil_mesh, pts_np,
                point_data={
                    "support_weights":   weight_full,
                    "spring_k_Npm3":    weight_full * winkler_k,
                },
            )
            written.append(mesh_path)

        return written

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
        geom: dict | None = None,
    ) -> tuple:
        """Traced V arrays for K_cs and K_sc coupling blocks.

        Parameters
        ----------
        curves_jax : list[CurveXYZFourierJAX]
        support_dofs : dict
        surface_pts_by_coil : list[jax.Array]
        geom : dict or None
            Pre-computed beam geometry (ignored for the uncoupled base support).

        Returns
        -------
        tuple (V_cs, V_sc) of empty jax.Arrays for the uncoupled base support.
        """
        return jnp.zeros(0), jnp.zeros(0)

    def coupling_terms(
        self,
        curves_jax,
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
        curves_jax : list[CurveXYZFourierJAX]
            Traced base-coil centreline objects.
        support_dofs : dict
            Merged support-dofs dict for the whole coil set.
        surface_pts_by_coil : list[jax.Array]
            Per-coil surface node positions, shape ``(n_surf_i, 3)`` each.
        coil_dof_offsets : list[int]
            DOF offset of each coil in the merged system.
        support_dof_offset : int
            DOF offset of the support block in the merged system.
        surface_node_indices_by_coil : list[np.ndarray]
            Per-coil integer arrays mapping surface-node compact index →
            global mesh node index.

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
