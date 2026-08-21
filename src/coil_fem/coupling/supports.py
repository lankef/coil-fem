"""Concrete grounded support and the base class for all support models.

Defines :class:`Support`, the simplest type of support structure. Coils are
fixed to stationary points in space via suspended clamps modelled with
Robin/Winkler boundary conditions.  Which surface nodes are clamped is
controlled by the optional ``fixed_clamp_fns`` callable(s) passed to
:class:`Support`.  Subclasses (e.g. :class:`~coil_fem.coupling.SupportBeams`)
extend the model with inter-coil coupling.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..pipelines import ElasticPipeline


@dataclasses.dataclass(frozen=True, eq=False)
class ContinuumMember:
    """Support-side continuum FEM domain (e.g. a central support ring).

    Published by :attr:`Support.continuum_members` so
    :class:`~coil_fem.CoilFEM` can include the member in structural metrics
    and VTU export without knowing the subclass.

    Parameters
    ----------
    name : str
        Short identifier used in VTU filenames (e.g. ``"csr"``).
    pipeline : ElasticPipeline
        Own mesh / problem / Lamé parameters.
    sym_weight : float
        Multiplier for extensive metrics (``l2_von_mises``, ``strain_energy``,
        …) so a one-period mesh represents the full symmetric structure.
        Peak metrics ignore this weight.
    mesh_points : callable
        ``mesh_points(support_dofs) -> (n_nodes, 3)``.
    solution : callable
        ``solution(u_s, support_dofs) -> sol_list`` with ``sol_list[0]`` of
        shape ``(n_nodes, 3)``.
    vtu_point_data : callable
        ``vtu_point_data(curves_jax, support_dofs, u_s) -> dict`` of
        numpy point-data arrays for VTU (may ignore ``u_s``).
    """

    name: str
    pipeline: ElasticPipeline
    sym_weight: float
    mesh_points: Callable
    solution: Callable
    vtu_point_data: Callable


class Support:
    """Grounded (fixed) support with a configurable Winkler weight function.

    Attachment points are spring-connected to a fixed ground; the effective
    support displacement seen by the coupling driver is always zero.  The
    spatial distribution of the Winkler spring stiffness is controlled by
    the optional ``fixed_clamp_fns`` argument.

    Pass an instance as ``CoilFEM(support=...)`` when no coupling to an
    external structural model is needed.  It corresponds to the Winkler /
    Robin BC built into :class:`~coil_fem.problems.LinearElasticity3D`.

    Subclasses
    ----------
    :class:`~coil_fem.coupling.SupportBeams`
        Bisymmetric beam-network model; solves a small linear beam problem
        and exposes coupled DOFs.

    Parameters
    ----------
    k_clamp : float
        Grounded Winkler spring stiffness [N/m³], applied at every surface
        quad point weighted by ``fixed_clamp_fns``.
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
        k_clamp: float,
        fixed_clamp_fns: Callable | list[Callable] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._k_clamp = float(k_clamp)
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
    def continuum_members(self) -> tuple[ContinuumMember, ...]:
        """Support-side continuum FEM domains for metrics and VTU.

        Empty for the grounded base support and for beam-only subclasses.
        Coupled subclasses that own a volumetric mesh (e.g.
        :class:`~coil_fem.coupling.SupportBeamsCSR`) override this.
        """
        return ()

    @property
    def matrix_symmetry(self) -> str:
        """``'symmetric'`` — unconditional by construction.

        Every spring field (grounded clamp, beam attachment) uses the same
        modulus on the coil side (:meth:`stiffness`) and the support side
        (:meth:`coupling_values` / :meth:`support_values`), so all support-block
        contributions (translational springs, moment-arm cross-terms) are
        symmetric.  :class:`SupportBeams` inherits this without an override.
        """
        return 'symmetric'

    @property
    def k_clamp(self) -> float:
        """Grounded Winkler spring modulus [N/m³]."""
        return self._k_clamp

    @property
    def k_attachment(self) -> float:
        """Distributed beam-attachment (Winkler) modulus [N/m³].

        ``0.0`` for the base (beam-less) support; coupled subclasses such as
        :class:`~coil_fem.coupling.SupportBeams` override this.
        """
        return 0.0

    def solve(self, inputs: dict) -> dict:
        """No-op; returns empty state dict."""
        return {}

    def beam_geometry(self, curves_jax: list, support_dofs: dict) -> dict | None:
        """Precomputed beam geometry bundle; ``None`` for uncoupled supports.

        Subclasses that benefit from single-pass geometry computation (e.g.
        :class:`~coil_fem.coupling.SupportBeams`) override this method to
        return a ``dict`` of traced arrays that can be threaded through
        :meth:`compute_weights`, :meth:`support_values`, and :meth:`coupling_values` via
        their ``geom`` keyword argument, avoiding redundant recomputation.

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
    ) -> tuple[jax.Array, jax.Array]:
        """Per-surface-node grounded-clamp and beam-attachment weights.

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
        w_g, w_a : jax.Array, shape ``(N,)`` each
            Grounded-clamp weight (in ``[0, 1]``, multiplies :attr:`k_clamp`)
            and beam-attachment weight (multiplies :attr:`k_attachment`,
            always zero here since the base support has no beams).
        """
        w_a = jnp.zeros(surface_pts.shape[0])
        if self._fixed_clamp_fns is None:
            return jnp.ones(surface_pts.shape[0]), w_a
        if isinstance(self._fixed_clamp_fns, list):
            fn = self._fixed_clamp_fns[coil_idx]
        else:
            fn = self._fixed_clamp_fns
        curve_i = curves_jax[coil_idx] if curves_jax is not None else None
        dofs_slice = (
            jax.tree_util.tree_map(lambda x: x[coil_idx], dofs)
            if dofs is not None else None
        )
        return fn(surface_pts, curve_i, dofs_slice), w_a

    def stiffness(self, w_g: jax.Array, w_a: jax.Array) -> jax.Array:
        """Per-point Winkler stiffness [N/m³]: ``k_clamp w_g + k_attachment w_a``."""
        return self._k_clamp * w_g + self.k_attachment * w_a

    def support_pattern(self):
        """Static local COO I/J index arrays for the support stiffness block ``K_ss``.

        Returns
        -------
        I, J : np.ndarray, shape ``(nnz,)``
            Row and column indices in support-local DOF numbering.  Empty
            int32 arrays for the uncoupled base support (no support DOFs).
        """
        import numpy as onp
        empty = onp.zeros(0, dtype=onp.int32)
        return empty, empty

    def support_values(
        self,
        curves_jax: list,
        support_dofs: dict,
        surface_pts_by_coil: list | None = None,
        geom: dict | None = None,
        *,
        jxw_by_coil: list | None = None,
        beam_endpoints=None,
    ):
        """Traced COO values for the support stiffness block ``K_ss``.

        Paired element-wise with :meth:`support_pattern`.  Override in
        subclasses that participate in monolithic assembly (e.g.
        :class:`~coil_fem.coupling.SupportBeams`).

        Parameters
        ----------
        curves_jax : list
            Differentiable centreline curves for all base coils.
        support_dofs : dict
            Optimisable support parameters.
        surface_pts_by_coil : list or None
            Per-coil surface query points.
        geom : dict or None
            Pre-computed geometry bundle from :meth:`beam_geometry`.
        jxw_by_coil : list or None
            Per-coil surface JxW area measures.
        beam_endpoints : list or None
            Pre-computed endpoint spring data (subclass-specific).

        Returns
        -------
        jax.Array, shape ``(nnz,)``
            Stiffness values ``K_ss[I[k], J[k]]`` [N/m].

        Raises
        ------
        NotImplementedError
            Default for the grounded :class:`Support`, which has no DOFs.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.support_values() is not implemented. "
            "Only supports that participate in monolithic assembly need this."
        )

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

