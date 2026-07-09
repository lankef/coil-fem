"""
Thin :class:`simsopt.Optimizable` wrapper around :class:`~coil_fem.CoilFEM`.

Magnetic field and equilibrium data stay **outside** the optimisable graph as
numpy constants; this module only connects coil geometry DOFs to coil_fem.
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
    """
    Simsopt :class:`~simsopt._core.optimizable.Optimizable` wrapping
    :class:`~coil_fem.CoilFEM`.

    Computes a weighted sum of FEM-based structural metrics over base coils and
    exposes :meth:`J` / :meth:`dJ` for use in simsopt optimisation loops.

    **How simsopt Optimizable objects work (brief primer)**

    Every :class:`~simsopt._core.optimizable.Optimizable` node stores degrees
    of freedom (DOFs) and participates in a DAG of objectives.

    * ``obj.x`` — flat numpy array of all **free** DOFs from ``obj`` *and all
      its ancestors* (the ``depends_on`` chain).  Ancestors are de-duplicated by
      the identity of their ``DOFs`` storage object, so shared parents (e.g.
      curves used by both ``Jforce`` and this objective) appear exactly *once*.
    * ``obj.x = dofs`` — distributes values back to each ancestor automatically.
    * ``obj.J()`` — scalar objective value.
    * ``obj.dJ()`` — flat gradient matching ``obj.x``.  The method body (before
      the ``@derivative_dec`` decoration) returns a
      :class:`~simsopt._core.derivative.Derivative` mapping
      ``{opt: grad_array}``; the decorator contracts it into a flat numpy array.
    * ``Derivative.__add__`` accumulates gradients for shared keys, so
      ``OptimizableSum`` (``JF + Jstress``) de-duplicates shared parents
      automatically.
    * ``recompute_bell()`` is called by simsopt whenever any ancestor's DOFs
      change; override it to invalidate caches.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (each exposing ``.curve`` and ``.current``).
        These are the *base* coils — before symmetry expansion.
    metrics : sequence of str
        Names of FEM metrics to include.  Available: ``'max_von_mises'``,
        ``'max_von_mises_lse'``, ``'mean_von_mises'``, ``'l2_von_mises'``,
        ``'strain_energy'``.
    metric_weights : sequence of float
        Weight applied to each metric entry.  Must be the same length as
        ``metrics``.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry during the symmetry expansion.
    mesh_options : dict or list[dict]
        Mesh construction options forwarded to
        :class:`~coil_fem.CoilFEM`.  See that class for details.
    material_options : dict or None
        Material properties forwarded to :class:`~coil_fem.CoilFEM`:
        ``'E'`` [Pa], ``'nu'``, ``'density'`` [kg/m³], and the optional thermal
        ``'itc'`` (positive integral thermal contraction ``ΔL/L`` applied
        as the eigenstrain ``ε_th = −itc · I``).
    base_supports : list[CoilSupport]
        Per-coil support models, same length as ``base_curves`` /
        ``base_currents`` (a single :class:`~coil_fem.simsopt.CoilSupport`
        is broadcast to every coil).  Each support contributes its own DOFs
        (e.g. clamp locations) to the optimisation graph, so support
        parameters are co-optimised with coil geometry and currents.
    problem_options : dict or None
        Options forwarded to the JAX-FEM problem constructor inside
        :class:`~coil_fem.CoilFEM`.
    gravity_options : dict or None
        Gravity body-force options forwarded to
        :class:`~coil_fem.CoilFEM`.  When ``None`` (default) no
        gravity load is applied.  When provided, may contain ``'g_vec'``
        (default ``(0, 0, -9.80665)``); the mass density is always taken
        from ``material_options['density']``.
    verbose : int
        JAX-FEM logging verbosity forwarded to :class:`~coil_fem.CoilFEM`,
        controlling output during construction, forward solves, and gradient
        evaluations:

        * ``0`` (default) — no logging (suppresses all JAX-FEM output).
        * ``1`` — INFO messages only (``[INFO]`` lines).
        * ``2`` — DEBUG messages too (full solver verbosity).

    Examples
    --------
    Drop-in addition to an existing simsopt optimisation loop::

        Jstress = CoilFEMObjective(
            base_coils,
            metrics=['max_von_mises_lse'],
            metric_weights=[1.0],
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            mesh_options={'shape': 'rect', 'w1': 0.02, 'w2': 0.02},
        )
        JTotal = JF + STRESS_WEIGHT * Jstress
        dofs = JTotal.x          # deduplicated; shared curves appear once

        def fun(dofs):
            JTotal.x = dofs
            return JTotal.J(), JTotal.dJ()

        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', ...)
    """

    def __init__(
        self,
        base_curves:list,
        base_currents: list,
        base_supports,
        metrics: Sequence[str],
        metric_weights: Sequence[float],
        nfp: int,
        stellsym: bool,
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
        from .support import CoilSupport

        if isinstance(metrics, str):
            metrics = [metrics]
        if isinstance(metric_weights, (int, float)):
            metric_weights = [metric_weights]

        if len(metrics) != len(metric_weights):
            raise ValueError(
                f"len(metrics)={len(metrics)} != "
                f"len(metric_weights)={len(metric_weights)}."
            )

        # ── Extract simsopt curve / current objects from the supplied coils ──
        assert len(base_curves) == len(base_currents)
        self._base_curves = base_curves
        self._base_currents = base_currents

        # Store scalar/dict constructor args as instance attributes so that
        # GSONable.as_dict can introspect them for JSON serialisation.
        self._nfp              = nfp
        self._stellsym         = stellsym
        self._mesh_options     = mesh_options
        self._material_options = material_options
        self._problem_options  = problem_options
        self._gravity_options  = gravity_options
        self._verbose          = verbose

        # ── Per-coil support models (broadcast a single one if given) ────────
        if isinstance(base_supports, CoilSupport):
            base_supports = [base_supports] * len(base_curves)
        if len(base_supports) != len(base_curves):
            raise ValueError(
                f"len(base_supports)={len(base_supports)} != "
                f"len(base_curves)={len(base_curves)}."
            )
        self._base_supports = list(base_supports)

        # ── Convert to JAX objects for CoilFEM construction ──────────────────
        base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c) for c in self._base_curves
        ]
        base_currents_jax = jnp.array(
            [c.get_value() for c in self._base_currents]
        )

        # ── Build the internal CoilFEM object (mesh topology fixed here) ─────
        self.fem = CoilFEM(
            base_curves_jax,
            base_currents_jax,
            [s.support_callable() for s in self._base_supports],
            [s.support_dofs for s in self._base_supports],
            nfp,
            stellsym,
            mesh_options,
            gravity_options=gravity_options,
            material_options=material_options,
            problem_options=problem_options,
            verbose=verbose,
        )

        self._metrics = tuple(metrics)
        self._metric_weights = list(metric_weights)

        # Caches: invalidated via recompute_bell() whenever DOFs change.
        # J and dJ are cached separately because the gradient (dJ) is far more
        # expensive than the forward value (J); calling J() must not trigger an
        # adjoint solve.
        self._needs_J: bool = True
        self._needs_dJ: bool = True
        self._J_cache: float | None = None
        self._grad_curves: list | None = None    # list[np.ndarray], one per base coil
        self._grad_currents: np.ndarray | None = None  # shape (n_base,)
        self._grad_supports: list | None = None  # list[dict], one per base coil

        # Register curve, current, and support Optimizables as parents so that
        # simsopt propagates DOF changes and de-duplicates x when combined with
        # other objectives (e.g. JF) that share the same parent objects.
        Optimizable.__init__(
            self,
            depends_on=(
                self._base_curves
                + self._base_currents
                + self._base_supports
            ),
        )

    # ── Cache invalidation ────────────────────────────────────────────────────

    def recompute_bell(self, child=None, parent=None):
        """Invalidate cached J / dJ when any ancestor DOFs change."""
        self._needs_J = True
        self._needs_dJ = True

    # ── Core computation ──────────────────────────────────────────────────────

    def _read_dofs(self):
        """Read coil / current / support DOFs live from the simsopt graph."""
        base_curves_dofs = [
            jnp.asarray(c.get_dofs()) for c in self._base_curves
        ]
        base_currents_dofs = jnp.array(
            [c.get_value() for c in self._base_currents]
        )
        # Support dofs are read live from each support's simsopt DOFs.
        base_support_dofs = [s.support_dofs for s in self._base_supports]
        return base_curves_dofs, base_currents_dofs, base_support_dofs

    def _weighted_J(self, cdofs, idofs, sdofs):
        """Weighted sum of requested FEM metrics (traced scalar)."""
        weights = self._metric_weights
        metrics = self._metrics
        result = self.fem.objective(cdofs, idofs, sdofs, metrics=metrics)
        return sum(w * result[m] for w, m in zip(weights, metrics))

    def _compute_J(self):
        """Evaluate only the (cheap) forward objective value.

        Does not trigger an adjoint solve, so calling :meth:`J` stays cheap.
        """
        if not self._needs_J:
            return
        base_curves_dofs, base_currents_dofs, base_support_dofs = self._read_dofs()
        self._J_cache = float(
            self._weighted_J(
                base_curves_dofs, base_currents_dofs, base_support_dofs
            )
        )
        self._needs_J = False

    def _compute_dJ(self):
        """Evaluate the (expensive) gradients, refreshing the J cache too.

        Uses :func:`jax.value_and_grad` so the objective value comes for free
        from the same AD pass; this also clears the J cache flag.
        """
        if not self._needs_dJ:
            return
        base_curves_dofs, base_currents_dofs, base_support_dofs = self._read_dofs()

        J_val, (grad_cdofs, grad_idofs, grad_sdofs) = value_and_grad(
            self._weighted_J, argnums=(0, 1, 2)
        )(base_curves_dofs, base_currents_dofs, base_support_dofs)

        self._J_cache = float(J_val)
        self._needs_J = False

        # Convert to numpy once so that simsopt Derivative assembly is cheap.
        self._grad_curves = [np.asarray(g) for g in grad_cdofs]
        self._grad_currents = np.asarray(grad_idofs)   # shape (n_base,)
        self._grad_supports = list(grad_sdofs)         # list[dict]
        self._needs_dJ = False

    # ── simsopt interface ─────────────────────────────────────────────────────

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

        # Assemble Derivative by mapping each JAX gradient array back to its
        # simsopt Optimizable.  Derivative.__add__ accumulates contributions for
        # any shared keys, so this is safe even when curves appear in multiple
        # objectives.
        d = Derivative({})
        for curve, g in zip(self._base_curves, self._grad_curves):
            # g has shape (curve.local_full_dof_size,) — same layout as
            # curve.get_dofs() since CurveXYZFourierJAX.from_simsopt uses
            # curve.get_dofs() directly.
            d = d + Derivative({curve: g})
        for current, g in zip(self._base_currents, self._grad_currents):
            # current.vjp handles ScaledCurrent / CurrentSum chain rules.
            d = d + current.vjp(np.array([float(g)]))
        for support, g in zip(self._base_supports, self._grad_supports):
            # g is a JAX dict; flatten_grad maps it to the support's flat DOFs.
            d = d + Derivative({support: support.flatten_grad(g)})
        return d

    return_fn_map = {'J': J, 'dJ': dJ}

    # ── Forward FEM ───────────────────────────────────────────────────────────

    def run(self):
        """Forward FEM for all base coils at the *current* simsopt DOFs.

        Reads coil geometry, currents, and support parameters live from the
        simsopt DOF graph and forwards them to
        :meth:`coil_fem.CoilFEM.run`, returning its full solution
        dict.  Intended for diagnostics, post-processing, and visualisation;
        no gradients are computed (use :meth:`J` / :meth:`dJ` for those).

        Returns
        -------
        dict
            See :meth:`coil_fem.CoilFEM.run` for the full set of keys
            (``'solutions'``, ``'displacements'``, ``'von_mises'``,
            ``'mesh_points'``, ``'support_weights'``, ``'f_vol'``,
            ``'B_self'``, ``'B_ext'``).
        """
        base_curves_dofs = [
            jnp.asarray(c.get_dofs()) for c in self._base_curves
        ]
        base_currents_dofs = jnp.array(
            [c.get_value() for c in self._base_currents]
        )
        base_support_dofs = [s.support_dofs for s in self._base_supports]
        return self.fem.run(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

    def save_run_vtu(self, out_dir: str = ".", *, prefix: str = "coil"):
        """Export per-coil forward-FEM results as VTU files at the *current* DOFs.

        Reads coil geometry, currents, and support parameters live from the
        simsopt DOF graph and forwards them to
        :meth:`coil_fem.CoilFEM.save_run_vtu`.  Because the support
        ``dofs`` are read from each support's :attr:`~coil_fem.simsopt.CoilSupport.support_dofs`,
        this works uniformly across support types (e.g. an empty ``{}`` for
        DOF-free supports such as
        :class:`~coil_fem.simsopt.CoilSupportTopBottom`, or ``{'phis': ...}``
        for :class:`~coil_fem.simsopt.CoilSupportDiscrete`).

        Parameters
        ----------
        out_dir : str
            Output directory.  Created if it does not exist.
        prefix : str
            File-name prefix (default ``"coil"``).

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        base_curves_dofs, base_currents_dofs, base_support_dofs = self._read_dofs()
        return self.fem.save_run_vtu(
            out_dir,
            prefix=prefix,
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

    def save_support_vtu(self, out_dir: str = ".", *, prefix: str = "coil"):
        """Export per-coil Winkler support weights as VTU files at the *current* DOFs.

        Reads coil geometry and support parameters live from the simsopt DOF
        graph and forwards them to
        :meth:`coil_fem.CoilFEM.save_support_vtu`.  Because the support
        ``dofs`` are read from each support's
        :attr:`~coil_fem.simsopt.CoilSupport.support_dofs`, this works
        uniformly across support types (e.g. an empty ``{}`` for DOF-free
        supports such as :class:`~coil_fem.simsopt.CoilSupportTopBottom`, or
        ``{'phis': ...}`` for :class:`~coil_fem.simsopt.CoilSupportDiscrete`).

        Parameters
        ----------
        out_dir : str
            Output directory.  Created if it does not exist.
        prefix : str
            File-name prefix (default ``"coil"``).

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        base_curves_dofs, _, base_support_dofs = self._read_dofs()
        return self.fem.save_support_vtu(
            out_dir,
            prefix=prefix,
            base_curves_dofs=base_curves_dofs,
            base_support_dofs=base_support_dofs,
        )

    def compute_strain_tensors(self):
        """Total and thermal strain tensors at the *current* simsopt DOFs.

        Reads coil geometry, currents, and support parameters live from the
        simsopt DOF graph and forwards them to
        :meth:`coil_fem.CoilFEM.compute_strain_tensors`.  Intended
        for diagnostics and post-processing; no gradients are computed.

        Returns
        -------
        dict
            See :meth:`coil_fem.CoilFEM.compute_strain_tensors` for
            the full set of keys (``'eps_total'``, ``'eps_thermal'``).
        """
        base_curves_dofs, base_currents_dofs, base_support_dofs = self._read_dofs()
        return self.fem.compute_strain_tensors(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        """The mesh nodes count."""
        return self.fem.n_nodes

    
    @property
    def n_cells(self) -> int:
        """The mesh cells count."""
        return self.fem.n_cells
    
    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_support(self, **kwargs):
        """Plot Winkler support weights at the *current* DOFs.

        Thin wrapper around :meth:`coil_fem.CoilFEM.plot_support`
        that feeds it the current coil geometry and support parameters read
        live from the simsopt DOF graph.  All keyword arguments are forwarded,
        including ``ax``, ``s``, ``cmap``, ``color``, ``simple_mode``, and any
        extra ``**kwargs`` passed through to :meth:`ax.scatter` (e.g.
        ``marker``, ``facecolors``, ``edgecolors``).  See
        :meth:`coil_fem.CoilFEM.plot_support` for the ``simple_mode`` semantics
        (support weight encoded as per-point alpha, no colorbar).

        Returns
        -------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            The 3-D axes used for the plot.  The parent figure is available as
            ``ax.get_figure()``.
        """

        base_curves_dofs = [
            jnp.asarray(c.get_dofs()) for c in self._base_curves
        ]
        base_support_dofs = [s.support_dofs for s in self._base_supports]
        return self.fem.plot_support(
            base_curves_dofs=base_curves_dofs,
            base_support_dofs=base_support_dofs,
            **kwargs,
        )

    def plot(self, engine: str = "matplotlib", ax=None, show: bool = True,
             axis_equal: bool = True, **kwargs):
        """Plot the von Mises stress surface over the support scatter.

        simsopt-compatible visualisation so this objective can be passed to
        :func:`simsopt.geo.plot` alongside coils and surfaces: that helper
        calls ``item.plot(ax=ax, show=..., **kwargs)`` on each item and expects
        the shared axes back.  This method reads coil geometry, currents, and
        support parameters live from the simsopt DOF graph and delegates to
        :meth:`coil_fem.CoilFEM.plot`.

        Parameters
        ----------
        engine : str
            Graphics engine.  Only ``"matplotlib"`` is supported.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None
            Existing 3-D axes to draw on (e.g. created by a previous item in
            :func:`simsopt.geo.plot`).  ``None`` creates a new figure/axes.
        show : bool
            Whether to call ``matplotlib.pyplot.show`` after plotting.  Set to
            ``False`` when more objects will be drawn on the same axes.
        axis_equal : bool
            Whether to scale the three axes equally.
        **kwargs
            Extra keyword arguments forwarded to
            :meth:`coil_fem.CoilFEM.plot` (e.g. ``cmap``, ``support_color``,
            ``support_s``).

        Returns
        -------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            The 3-D axes used for the plot, so it can be forwarded to the next
            item in :func:`simsopt.geo.plot`.
        """
        if engine != "matplotlib":
            raise NotImplementedError(
                "CoilFEMObjective.plot supports the matplotlib engine only."
            )

        base_curves_dofs, base_currents_dofs, base_support_dofs = self._read_dofs()
        ax = self.fem.plot(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
            ax=ax,
            axis_equal=axis_equal,
            **kwargs,
        )
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax
