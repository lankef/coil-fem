"""Simsopt ``Optimizable`` wrapper around :class:`~coil_fem.CoilFEM`.

Connects coil geometry DOFs to the structural FEM pipeline via
:class:`CoilFEMObjective`, exposing :meth:`~CoilFEMObjective.J` and
:meth:`~CoilFEMObjective.dJ` for use in simsopt optimisation loops.
A single :class:`~coil_fem.simsopt.CoilSupport` object is the one entry
point: it holds the base coils (curves + currents), ``nfp``, ``stellsym``,
and any optimisable support DOFs (e.g. clamp locations).
"""

from __future__ import annotations

from typing import Sequence
from jax import value_and_grad
import numpy as np
import jax.numpy as jnp

try:
    from simsopt._core.optimizable import Optimizable
    from simsopt._core.derivative import derivative_dec, Derivative
    _HAS_SIMSOPT = True
except ImportError:  # pragma: no cover
    Optimizable = object           # type: ignore[misc, assignment]
    _HAS_SIMSOPT = False

    def derivative_dec(fn):        # type: ignore[misc]
        return fn

    class Derivative:              # type: ignore[no-redef]
        def __init__(self, d):
            self.d = d


class CoilFEMObjective(Optimizable):
    """Simsopt ``Optimizable`` wrapping :class:`~coil_fem.CoilFEM`.

    Computes a weighted sum of FEM structural metrics and exposes :meth:`J`
    / :meth:`dJ` for use in simsopt optimisation loops.  All coil and
    support data come from a single ``coil_support`` object.

    Parameters
    ----------
    coil_support : CoilSupport
        Holds the base coils (curves + currents), ``nfp``, ``stellsym``, and
        any optimisable support DOFs.  It is the only ``depends_on`` entry
        registered with simsopt; curves and currents are reached through it.
    metrics : sequence of str
        Names of FEM metrics to include.  Available: ``'max_von_mises'``,
        ``'max_von_mises_lse'``, ``'mean_von_mises'``, ``'l2_von_mises'``,
        ``'strain_energy'``.
    metric_weights : sequence of float
        Weight applied to each metric.  Must have the same length as
        ``metrics``.
    mesh_options : dict or list[dict]
        Mesh construction options forwarded to :class:`~coil_fem.CoilFEM`.
    material_options : dict or None
        Material properties (``'E'``, ``'nu'``, ``'density'``, ``'itc'``).
    problem_options : dict or None
        Solver options forwarded to :class:`~coil_fem.CoilFEM`.
    gravity_options : dict or None
        Gravity body-force options forwarded to :class:`~coil_fem.CoilFEM`.
    verbose : int
        JAX-FEM logging verbosity (0 = silent, 1 = INFO, 2 = DEBUG).

    Examples
    --------
    Drop-in addition to an existing simsopt optimisation loop::

        coil_support = CoilSupportFixed(
            base_coils,
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            clamp_radius=0.05,
        )
        Jstress = CoilFEMObjective(
            coil_support,
            metrics=['max_von_mises_lse'],
            metric_weights=[1.0],
            mesh_options={'shape': 'rect', 'w1': 0.02, 'w2': 0.02},
            problem_options={'winkler_k': 1e9},
        )
        JTotal = JF + STRESS_WEIGHT * Jstress

        def fun(dofs):
            JTotal.x = dofs
            return JTotal.J(), JTotal.dJ()
    """

    def __init__(
        self,
        coil_support,
        metrics: Sequence[str],
        metric_weights: Sequence[float],
        mesh_options,
        material_options=None,
        problem_options=None,
        gravity_options=None,
        verbose: int = 0,
    ):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for CoilFEMObjective.")

        from ..geo import CurveXYZFourierJAX
        from ..coil_fem import CoilFEM

        if isinstance(metrics, str):
            metrics = [metrics]
        if isinstance(metric_weights, (int, float)):
            metric_weights = [metric_weights]

        if len(metrics) != len(metric_weights):
            raise ValueError(
                f"len(metrics)={len(metrics)} != "
                f"len(metric_weights)={len(metric_weights)}."
            )

        self._coil_support = coil_support

        # Store constructor args for serialisation introspection.
        self._mesh_options     = mesh_options
        self._material_options = material_options
        self._problem_options  = problem_options
        self._gravity_options  = gravity_options
        self._verbose          = verbose

        # ============================================================================
        # Build JAX coil objects from coil_support
        # ============================================================================
        base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c) for c in coil_support.base_curves
        ]
        base_currents_jax = jnp.array(
            [c.get_value() for c in coil_support.base_currents]
        )

        # ============================================================================
        # Build CoilFEM (mesh topology fixed here)
        # ============================================================================
        self.fem = CoilFEM(
            base_curves_jax,
            base_currents_jax,
            coil_support.nfp,
            coil_support.stellsym,
            mesh_options,
            support=coil_support,
            gravity_options=gravity_options,
            material_options=material_options,
            problem_options=problem_options,
            verbose=verbose,
        )

        self._metrics = tuple(metrics)
        self._metric_weights = list(metric_weights)

        # Caches invalidated via recompute_bell() when any DOFs change.
        self._needs_J: bool = True
        self._needs_dJ: bool = True
        self._J_cache: float | None = None
        self._grad_curves: list | None = None
        self._grad_currents: np.ndarray | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support])

    # ============================================================================
    # Cache invalidation
    # ============================================================================

    def recompute_bell(self, child=None, parent=None):
        """Invalidate cached J / dJ when any ancestor DOFs change."""
        self._needs_J = True
        self._needs_dJ = True

    # ============================================================================
    # Core computation
    # ============================================================================

    def _read_dofs(self):
        """Read coil / current / support DOFs live from the simsopt graph."""
        base_curves_dofs = [
            jnp.asarray(c.get_dofs())
            for c in self._coil_support.base_curves
        ]
        base_currents_dofs = jnp.array(
            [c.get_value() for c in self._coil_support.base_currents]
        )
        support_dofs = self._coil_support.support_dofs
        return base_curves_dofs, base_currents_dofs, support_dofs

    def _weighted_J(self, cdofs, idofs, sdofs):
        """Weighted sum of requested FEM metrics (traced scalar)."""
        result = self.fem.objective(cdofs, idofs, sdofs, metrics=self._metrics)
        return sum(w * result[m] for w, m in zip(self._metric_weights, self._metrics))

    def _compute_J(self):
        """Evaluate the forward objective value without an adjoint solve."""
        if not self._needs_J:
            return
        cdofs, idofs, sdofs = self._read_dofs()
        self._J_cache = float(self._weighted_J(cdofs, idofs, sdofs))
        self._needs_J = False

    def _compute_dJ(self):
        """Evaluate gradients (and refresh J cache) via value_and_grad."""
        if not self._needs_dJ:
            return
        cdofs, idofs, sdofs = self._read_dofs()

        J_val, (grad_cdofs, grad_idofs, grad_sdofs) = value_and_grad(
            self._weighted_J, argnums=(0, 1, 2)
        )(cdofs, idofs, sdofs)

        self._J_cache = float(J_val)
        self._needs_J = False

        self._grad_curves   = [np.asarray(g) for g in grad_cdofs]
        self._grad_currents = np.asarray(grad_idofs)
        self._grad_support  = grad_sdofs   # single dict
        self._needs_dJ = False

    # ============================================================================
    # Simsopt interface
    # ============================================================================

    def J(self):
        """Weighted sum of FEM metrics (scalar)."""
        self._compute_J()
        return self._J_cache

    @derivative_dec
    def dJ(self):
        """Gradient of J w.r.t. all free DOFs in the graph.

        Returns a :class:`~simsopt._core.derivative.Derivative` object.
        ``@derivative_dec`` contracts it into a flat numpy array aligned with
        ``self.x`` before returning to the caller.
        """
        self._compute_dJ()

        d = Derivative({})
        for curve, g in zip(self._coil_support.base_curves, self._grad_curves):
            d = d + Derivative({curve: g})
        for current, g in zip(self._coil_support.base_currents, self._grad_currents):
            d = d + current.vjp(np.array([float(g)]))
        d = d + Derivative({
            self._coil_support:
                self._coil_support.flatten_grad(self._grad_support)
        })
        return d

    return_fn_map = {'J': J, 'dJ': dJ}

    # ============================================================================
    # Forward FEM helpers
    # ============================================================================

    def run(self):
        """Forward FEM for all base coils at the *current* simsopt DOFs.

        Returns
        -------
        dict
            See :meth:`coil_fem.CoilFEM.run`.
        """
        cdofs, idofs, sdofs = self._read_dofs()
        return self.fem.run(
            base_curves_dofs=cdofs,
            base_currents_dofs=idofs,
            base_support_dofs=sdofs,
        )

    def save_run_vtu(self, out_dir: str = ".", *, prefix: str = "coil"):
        """Export per-coil FEM results as VTU files at the *current* DOFs.

        Parameters
        ----------
        out_dir : str
            Output directory.
        prefix : str
            File-name prefix.

        Returns
        -------
        list[str]
            Paths of all files written.
        """
        cdofs, idofs, sdofs = self._read_dofs()
        return self.fem.save_run_vtu(
            out_dir,
            prefix=prefix,
            base_curves_dofs=cdofs,
            base_currents_dofs=idofs,
            base_support_dofs=sdofs,
        )

    def save_support_vtu(self, out_dir: str = ".", *, prefix: str = "coil"):
        """Export per-coil Winkler support weights as VTU files at the *current* DOFs.

        Parameters
        ----------
        out_dir : str
            Output directory.
        prefix : str
            File-name prefix.

        Returns
        -------
        list[str]
            Paths of all files written.
        """
        cdofs, _, sdofs = self._read_dofs()
        return self.fem.save_support_vtu(
            out_dir,
            prefix=prefix,
            base_curves_dofs=cdofs,
            base_support_dofs=sdofs,
        )

    def compute_strain_tensors(self):
        """Total and thermal strain tensors at the *current* simsopt DOFs.

        Returns
        -------
        dict
            See :meth:`coil_fem.CoilFEM.compute_strain_tensors`.
        """
        cdofs, idofs, sdofs = self._read_dofs()
        return self.fem.compute_strain_tensors(
            base_curves_dofs=cdofs,
            base_currents_dofs=idofs,
            base_support_dofs=sdofs,
        )

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def n_nodes(self) -> int:
        """The mesh node count."""
        return self.fem.n_nodes

    @property
    def n_cells(self) -> int:
        """The mesh cell count."""
        return self.fem.n_cells

    # ============================================================================
    # Visualisation
    # ============================================================================

    def plot_support(self, **kwargs):
        """Plot Winkler support weights at the *current* DOFs.

        Thin wrapper around :meth:`coil_fem.CoilFEM.plot_support`.  All
        keyword arguments are forwarded.

        Returns
        -------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        """
        cdofs, _, sdofs = self._read_dofs()
        return self.fem.plot_support(
            base_curves_dofs=cdofs,
            base_support_dofs=sdofs,
            **kwargs,
        )

    def plot(self, engine: str = "matplotlib", ax=None, show: bool = True,
             axis_equal: bool = True, **kwargs):
        """Plot von Mises stress surface over the support scatter.

        Parameters
        ----------
        engine : str
            Graphics engine (only ``"matplotlib"`` supported).
        ax : Axes3D or None
        show : bool
        axis_equal : bool
        **kwargs
            Forwarded to :meth:`coil_fem.CoilFEM.plot`.

        Returns
        -------
        ax : Axes3D
        """
        if engine != "matplotlib":
            raise NotImplementedError(
                "CoilFEMObjective.plot supports the matplotlib engine only."
            )

        cdofs, idofs, sdofs = self._read_dofs()
        ax = self.fem.plot(
            base_curves_dofs=cdofs,
            base_currents_dofs=idofs,
            base_support_dofs=sdofs,
            ax=ax,
            axis_equal=axis_equal,
            **kwargs,
        )
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax
