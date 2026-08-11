"""Simsopt ``Optimizable`` wrapper around :class:`~coil_fem.CoilFEM`.

Connects coil geometry DOFs to the structural FEM pipeline via
:class:`CoilFEMObjective`, exposing :meth:`~CoilFEMObjective.J` and
:meth:`~CoilFEMObjective.dJ` for use in simsopt optimisation loops.
A single :class:`~coil_fem.simsopt.CoilSupport` object is the one entry
point: it holds the base coils (curves + currents), ``nfp``, ``stellsym``,
and any optimisable support DOFs (e.g. clamp locations).
:class:`BeamSurfaceDistance` is a geometric companion penalty that keeps
support beams clear of a target surface.
:class:`BeamCurveDistance` hinges free-span beam clearance to the attached
coil curves.
:class:`BeamCurveAngle` penalises beam–coil attachments that are too nearly
tangent.
"""

from __future__ import annotations

from typing import Sequence
import math
import jax
from jax import value_and_grad
import numpy as np
import jax.numpy as jnp

from ..geo import CurveXYZFourierJAX
from ..problems import recompute_fe_geometry
from ..metrics import total_strain_energy

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
        Solver options forwarded to :class:`~coil_fem.CoilFEM`
        (including ``'remat_bs'``, default True).
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
            fixed_clamp_options={
                'k_clamp': 1e9,
                'r_clamp': 0.05,
                'n_clamp': 2,
            },
        )
        Jstress = CoilFEMObjective(
            coil_support,
            metrics=['max_von_mises_lse'],
            metric_weights=[1.0],
            mesh_options={'shape': 'rect', 'w1': 0.02, 'w2': 0.02},
            problem_options={'solver': 'umfpack'},
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
        physics_options=None,
        coupling='monolithic',
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
        self._physics_options  = physics_options
        self._coupling         = coupling
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
            support=coil_support.support,
            gravity_options=gravity_options,
            material_options=material_options,
            problem_options=problem_options,
            physics_options=physics_options,
            coupling=coupling,
            verbose=verbose,
        )

        self._metrics = tuple(metrics)
        self._metric_weights = list(metric_weights)

        # On the cuDSS path the full value_and_grad computation is JIT-able:
        # merged_solve is wrapped with custom_vjp (GPU FFI), set_params writes
        # and reads happen within the same trace, and mesh shapes are fixed at
        # construction.  Cache the compiled function so subsequent calls avoid
        # re-tracing.  On the CPU/staggered path the Newton loop contains
        # host syncs (float conversions) so JIT is not applied.
        _use_jit = (
            problem_options is not None
            and problem_options.get('solver', 'umfpack') == 'cudss'
        )
        _vg = value_and_grad(self._weighted_J, argnums=(0, 1, 2))
        if _use_jit:
            self._jit_vg: object = jax.jit(_vg)
        else:
            self._jit_vg = _vg

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
        J_val, _ = self._jit_vg(cdofs, idofs, sdofs)
        self._J_cache = float(J_val)
        self._needs_J = False

    def _compute_dJ(self):
        """Evaluate gradients (and refresh J cache) via value_and_grad."""
        if not self._needs_dJ:
            return
        cdofs, idofs, sdofs = self._read_dofs()

        J_val, (grad_cdofs, grad_idofs, grad_sdofs) = self._jit_vg(
            cdofs, idofs, sdofs
        )

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

    def summary(self):
        """RMS/max displacement, RMS/max von Mises, and total strain energy.
    
        von Mises and the body-force magnitude are ``(n_cells, n_quads)`` quadrature
        fields, weighted directly by JxW.  Displacement magnitude is a nodal field
        interpolated to the quadrature points with the element shape functions before
        weighting.  Maxima are taken over the raw (unsmoothed) quantities.
        """
        result = self.run()
        fem = self.fem
        lam, mu = fem._lam, fem._mu
        num_d2 = num_vm2 = num_f2 = vol = 0.0
        max_d = max_vm = max_f = 0.0
        strain_energy = 0.0
        for i in range(len(result['von_mises'])):
            prob = fem.pipelines[i].problem
            pts  = result['mesh_points'][i]
            sg, jxw_j, _, _ = recompute_fe_geometry(
                pts, prob._cells_jnp, prob._sg_ref, prob._sv, prob._qw)
            jxw   = np.asarray(jxw_j)               # (n_cells, n_quads)
            sv    = np.asarray(prob._sv)            # (n_quads, n_cell_nodes)
            cells = np.asarray(prob._cells_jnp)     # (n_cells, n_cell_nodes)
    
            # von Mises: quadrature field -> weight by JxW directly
            vm = np.asarray(result['von_mises'][i])
            num_vm2 += np.sum(vm**2 * jxw)
            max_vm   = max(max_vm, float(np.max(vm)))
    
            # |u|: nodal field -> interpolate to quadrature points, then weight
            dmag   = np.linalg.norm(np.asarray(result['displacements'][i]), axis=-1)
            dmag_q = np.einsum('qn,cn->cq', sv, dmag[cells])
            num_d2 += np.sum(dmag_q**2 * jxw)
            max_d   = max(max_d, float(np.max(dmag)))
    
            # body force: quadrature field magnitude -> weight by JxW directly
            fmag = np.linalg.norm(np.asarray(result['f_vol'][i]), axis=-1)
            num_f2 += np.sum(fmag**2 * jxw)
            max_f   = max(max_f, float(np.max(fmag)))
    
            vol += np.sum(jxw)
            strain_energy += float(total_strain_energy(
                prob, result['solutions'][i], lam, mu, shape_grads=sg, JxW=jxw_j))
        return {
            'rms_displacement_m': float(np.sqrt(num_d2 / vol)),
            'max_displacement_m': max_d,
            'rms_von_mises_Pa':   float(np.sqrt(num_vm2 / vol)),
            'max_von_mises_Pa':   max_vm,
            'rms_body_force_Npm3': float(np.sqrt(num_f2 / vol)),
            'max_body_force_Npm3': max_f,
            'strain_energy_J':    strain_energy,
        }

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


# ============================================================================
# Beam-surface distance
# ============================================================================


def _segment_point_dists(x_start, x_end, pts):
    """Distances from points to line segments.

    Parameters
    ----------
    x_start : jax.Array, shape (N, 3)
        Segment start points.
    x_end : jax.Array, shape (N, 3)
        Segment end points.
    pts : jax.Array, shape (M, 3)
        Query points.

    Returns
    -------
    jax.Array, shape (N, M)
        ``dists[i, j]`` is the distance from ``pts[j]`` to segment ``i``.
    """
    d_vec = x_end - x_start                                     # (N, 3)
    w = pts[None, :, :] - x_start[:, None, :]                   # (N, M, 3)
    # Chord station of the closest point, clamped to the segment.
    t = jnp.clip(
        jnp.sum(w * d_vec[:, None, :], axis=2)
        / (jnp.sum(d_vec * d_vec, axis=1)[:, None] + 1e-300),
        0.0,
        1.0,
    )                                                           # (N, M)
    delta = w - t[:, :, None] * d_vec[:, None, :]               # (N, M, 3)
    return jnp.sqrt(jnp.sum(delta ** 2, axis=2))


def _beam_surface_distance_pure(x_start, x_end, L, gammas, ns, minimum_distance):
    r"""Hinge penalty on beam-chord-to-surface distance.

    .. math::
        J = \left\langle L_b \, \|\mathbf{n}_s\| \,
            \max(0, d_\min - d_{bs})^2 \right\rangle_{b, s}

    Parameters
    ----------
    x_start : jax.Array, shape (N, 3)
        Beam endpoints at node 1.
    x_end : jax.Array, shape (N, 3)
        Beam endpoints at node 2.
    L : jax.Array, shape (N,)
        Beam chord lengths; the arclength element of a chord on ``[0, 1]``.
    gammas : jax.Array, shape (M, 3)
        Surface quadrature points.
    ns : jax.Array, shape (M, 3)
        Unnormalised surface normals; the magnitude is the area element.
    minimum_distance : float
        Threshold below which the penalty activates.

    Returns
    -------
    jax.Array, shape ()
        Scalar penalty value.
    """
    dists = _segment_point_dists(x_start, x_end, gammas)
    integralweight = L[:, None] * jnp.linalg.norm(ns, axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance - dists, 0) ** 2)


class BeamSurfaceDistance(Optimizable):
    r"""Penalise support beams that come closer than ``minimum_distance`` to a surface.

    The beam analogue of :class:`simsopt.geo.CurveSurfaceDistance`: the same
    hinge form and scaling, so the two terms share a weight scale.  Support
    beams are straight chords rather than sampled curves, so distances use the
    exact point-to-segment formula instead of a quadrature over beam points.
    All coil-coil and coil-foundation beams contribute.

    .. math::
        J = \left\langle L_b \, \|\mathbf{n}_s\| \,
            \max(0, d_\min - d_{bs})^2 \right\rangle_{b, s}

    where :math:`d_{bs}` is the distance from surface point :math:`s` to beam
    chord :math:`b` and :math:`L_b` is the chord length.

    Parameters
    ----------
    coil_support_beams : CoilSupportBeams
        Provides the base curves, the beam DOFs, and the underlying
        :class:`~coil_fem.coupling.SupportBeams` model.
    surface : simsopt.geo.Surface
        Target surface (typically the plasma boundary).  It is *not* a DOF
        parent, so it contributes no gradient — matching
        :class:`simsopt.geo.CurveSurfaceDistance`.
    minimum_distance : float
        Desired minimum beam-to-surface clearance [m].

    Notes
    -----
    Only the base (master) beams held in ``SupportBeams.beam_geometry`` are
    summed; stellarator-mirrored partners and field-period rotations are not
    replicated.  For a symmetric surface those images have identical distances,
    so the omitted symmetry factor is absorbed into the objective weight.

    The full centreline chord ``x_start -> x_end`` is used, not the free span
    between ``xi_start`` and ``xi_end``.

    Examples
    --------
    >>> Jbeam = BeamSurfaceDistance(coil_support, surface, minimum_distance=0.15)
    >>> Jbeam.shortest_distance()  # doctest: +SKIP
    0.2731...
    """

    def __init__(self, coil_support_beams, surface, minimum_distance: float):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for BeamSurfaceDistance.")

        self._coil_support = coil_support_beams
        self._support = coil_support_beams.support
        self.surface = surface
        self.minimum_distance = float(minimum_distance)

        # Reference curves supply the (static) quadpoints and order; the DOFs
        # are re-read live so the curve path stays differentiable.
        self._base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c)
            for c in coil_support_beams.base_curves
        ]

        self._jit_vg = jax.jit(value_and_grad(self._J_pure, argnums=(0, 1)))

        # Cache invalidated via recompute_bell() when any DOFs change.
        self._needs_update: bool = True
        self._J_cache: float | None = None
        self._grad_curves: list | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support_beams])

    # ============================================================================
    # Cache invalidation
    # ============================================================================

    def recompute_bell(self, child=None, parent=None):
        """Invalidate cached J / dJ when any ancestor DOFs change."""
        self._needs_update = True

    # ============================================================================
    # Core computation
    # ============================================================================

    def _read_dofs(self):
        """Read coil / support DOFs live from the simsopt graph."""
        base_curves_dofs = [
            jnp.asarray(c.get_dofs())
            for c in self._coil_support.base_curves
        ]
        return base_curves_dofs, self._coil_support.support_dofs

    def _surface_arrays(self):
        """Surface quadrature points and unnormalised normals, both ``(M, 3)``."""
        return (
            jnp.asarray(self.surface.gamma().reshape((-1, 3))),
            jnp.asarray(self.surface.normal().reshape((-1, 3))),
        )

    def _beam_chords(self, cdofs, sdofs):
        """Beam chord ``(x_start, x_end, L)`` for the given DOFs (traced)."""
        curves_jax = [
            CurveXYZFourierJAX(ref.quadpoints, d, ref.order)
            for ref, d in zip(self._base_curves_jax, cdofs)
        ]
        # SupportBeams caches nothing, so this recomputes the full geom dict;
        # only these three of its ten entries are used.
        geom = self._support.beam_geometry(curves_jax, sdofs)
        return geom['x_start'], geom['x_end'], geom['L']

    def _J_pure(self, cdofs, sdofs, gammas, ns):
        """Beam-to-surface hinge penalty (traced scalar)."""
        x_start, x_end, L = self._beam_chords(cdofs, sdofs)
        return _beam_surface_distance_pure(
            x_start, x_end, L, gammas, ns, self.minimum_distance,
        )

    def _compute(self):
        """Evaluate J and its gradients from the single ``value_and_grad``."""
        if not self._needs_update:
            return
        cdofs, sdofs = self._read_dofs()
        gammas, ns = self._surface_arrays()

        J_val, (grad_cdofs, grad_sdofs) = self._jit_vg(cdofs, sdofs, gammas, ns)

        self._J_cache = float(J_val)
        self._grad_curves = [np.asarray(g) for g in grad_cdofs]
        self._grad_support = grad_sdofs
        self._needs_update = False

    # ============================================================================
    # Simsopt interface
    # ============================================================================

    def J(self):
        """Beam-to-surface hinge penalty (scalar)."""
        self._compute()
        return self._J_cache

    @derivative_dec
    def dJ(self):
        """Gradient of J w.r.t. the coil and support DOFs.

        Returns a :class:`~simsopt._core.derivative.Derivative` object.
        ``@derivative_dec`` contracts it into a flat numpy array aligned with
        ``self.x`` before returning to the caller.
        """
        self._compute()

        d = Derivative({})
        for curve, g in zip(self._coil_support.base_curves, self._grad_curves):
            d = d + Derivative({curve: g})
        return d + Derivative({
            self._coil_support:
                self._coil_support.flatten_grad(self._grad_support)
        })

    return_fn_map = {'J': J, 'dJ': dJ}

    def shortest_distance(self):
        """Smallest beam-chord-to-surface-point distance [m].

        Returns
        -------
        float
            Diagnostic clearance; the penalty is zero when this exceeds
            ``minimum_distance``.
        """
        cdofs, sdofs = self._read_dofs()
        x_start, x_end, _ = self._beam_chords(cdofs, sdofs)
        gammas, _ = self._surface_arrays()
        return float(jnp.min(_segment_point_dists(x_start, x_end, gammas)))


# ============================================================================
# Beam-curve distance
# ============================================================================


def _curve_segment_hinge(x_a, x_b, gamma, gammadash, minimum_distance):
    r"""Per-beam arclength-weighted hinge of segment-to-curve distance.

    .. math::
        J_b = \bigl\langle
            \|\gamma'\| \,
            \max(0,\, d_\min - d_b)^2
        \bigr\rangle_{\phi}

    where :math:`d_b` is the distance from the curve sample to segment
    ``x_a[b] -> x_b[b]``.

    Parameters
    ----------
    x_a, x_b : jax.Array, shape (N, 3)
        Effective free-span endpoints.
    gamma : jax.Array, shape (M, 3)
        Curve quadrature points.
    gammadash : jax.Array, shape (M, 3)
        Curve tangents ``γ'`` (``dγ/dφ``); ``‖γ'‖`` weights the quadrature.
    minimum_distance : float
        Threshold below which the penalty activates.

    Returns
    -------
    jax.Array, shape (N,)
        Per-beam mean hinge values.
    """
    dists = _segment_point_dists(x_a, x_b, gamma)
    alen = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(
        alen[None, :] * jnp.maximum(minimum_distance - dists, 0.0) ** 2,
        axis=1,
    )


class BeamCurveDistance(Optimizable):
    r"""Penalise support beams that come closer than ``minimum_distance`` to coils.

    The beam analogue of :class:`simsopt.geo.CurveCurveDistance`: the same
    hinge form, with distances measured from each coil curve to the beam's
    effective free-span segment rather than between two sampled curves.

    .. math::
        J = \sum_b \Biggl(
            \int_{\gamma_b^\mathrm{start}}
                \max\bigl(0,\, d_\min - d_b^\mathrm{start}\bigr)^2
                \, dl^\mathrm{start}
            + \int_{\gamma_b^\mathrm{end}}
                \max\bigl(0,\, d_\min - d_b^\mathrm{end}\bigr)^2
                \, dl^\mathrm{end}
        \Biggr)

    where :math:`d_b^\mathrm{start}` (:math:`d_b^\mathrm{end}`) is the
    Euclidean distance from a point on the start (end) coil to the free-span
    segment :math:`S_b` of beam :math:`b`.  Coil–foundation (CF) beams omit
    the end integral.  The free span is the chord station interval

    .. math::
        \xi_\mathrm{start}^\mathrm{eff}
            = \max(\xi_\mathrm{start},\, r_\mathrm{safe}/L),
        \qquad
        \xi_\mathrm{end}^\mathrm{eff}
            = \min(\xi_\mathrm{end},\, 1 - r_\mathrm{safe}/L),

    with :math:`S_b` the segment between those stations.  When
    :math:`\xi_\mathrm{start}^\mathrm{eff} > \xi_\mathrm{end}^\mathrm{eff}`,
    beam :math:`b` contributes zero.

    Parameters
    ----------
    coil_support_beams : CoilSupportBeams
        Provides the base curves, the beam DOFs, and the underlying
        :class:`~coil_fem.coupling.SupportBeams` model.
    dead_length : float
        Length ignored from each chord end before the free span [m].
    minimum_distance : float
        Desired minimum beam-to-coil clearance [m].

    Notes
    -----
    Only the base (master) beams held in ``SupportBeams.beam_geometry`` are
    summed; stellarator-mirrored partners and field-period rotations are not
    replicated.

    Examples
    --------
    >>> Jbc = BeamCurveDistance(coil_support, dead_length=0.05, minimum_distance=0.1)
    >>> Jbc.shortest_distance()  # doctest: +SKIP
    0.18...
    """

    def __init__(
        self,
        coil_support_beams,
        dead_length: float,
        minimum_distance: float,
    ):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for BeamCurveDistance.")

        dead_length = float(dead_length)
        minimum_distance = float(minimum_distance)
        if dead_length < 0.0:
            raise ValueError(f"dead_length must be >= 0; got {dead_length}.")
        if minimum_distance < 0.0:
            raise ValueError(
                f"minimum_distance must be >= 0; got {minimum_distance}."
            )

        self._coil_support = coil_support_beams
        self._support = coil_support_beams.support
        self.dead_length = dead_length
        self.minimum_distance = minimum_distance

        self._base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c)
            for c in coil_support_beams.base_curves
        ]

        self._jit_vg = jax.jit(value_and_grad(self._J_pure, argnums=(0, 1)))

        self._needs_update: bool = True
        self._J_cache: float | None = None
        self._grad_curves: list | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support_beams])

    def recompute_bell(self, child=None, parent=None):
        """Invalidate cached J / dJ when any ancestor DOFs change."""
        self._needs_update = True

    def _read_dofs(self):
        """Read coil / support DOFs live from the simsopt graph."""
        base_curves_dofs = [
            jnp.asarray(c.get_dofs())
            for c in self._coil_support.base_curves
        ]
        return base_curves_dofs, self._coil_support.support_dofs

    def _curves_jax(self, cdofs):
        """Rebuild traced curve pytrees from live DOFs."""
        return [
            CurveXYZFourierJAX(ref.quadpoints, d, ref.order)
            for ref, d in zip(self._base_curves_jax, cdofs)
        ]

    def _effective_segments(self, geom):
        """Free-span endpoints ``(x_a, x_b)`` and active mask from ``geom``."""
        x_start = geom['x_start']
        x_end = geom['x_end']
        L = geom['L']
        xi_safe_start = self.dead_length / (L + 1e-300)
        xi_safe_end = 1.0 - self.dead_length / (L + 1e-300)
        xi_start_eff = jnp.maximum(geom['xi_start'], xi_safe_start)
        xi_end_eff = jnp.minimum(geom['xi_end'], xi_safe_end)
        active = xi_start_eff <= xi_end_eff
        d = x_end - x_start
        x_a = x_start + xi_start_eff[:, None] * d
        x_b = x_start + xi_end_eff[:, None] * d
        return x_a, x_b, active

    def _accumulate_J(self, curves_jax, x_a, x_b, active):
        """Sum start/end hinges over CC groups and start-only hinges over CF."""
        support = self._support
        dmin = self.minimum_distance
        J = jnp.array(0.0)
        b = 0
        n_base = support.n_base

        def add_cc(J0, g, b0):
            n_g = support.n_beam_cc[g]
            if n_g == 0:
                return J0, b0
            start_idx, end_idx, end_tfm = support.cc_groups[g]
            sl = slice(b0, b0 + n_g)
            c_s = curves_jax[start_idx]
            c_e = curves_jax[end_idx]
            gamma_e = support._apply_end_transform(c_e.gamma(), end_tfm)
            hs = _curve_segment_hinge(
                x_a[sl], x_b[sl], c_s.gamma(), c_s.gammadash(), dmin,
            )
            he = _curve_segment_hinge(
                x_a[sl], x_b[sl], gamma_e, c_e.gammadash(), dmin,
            )
            return J0 + jnp.sum(jnp.where(active[sl], hs + he, 0.0)), b0 + n_g

        for i in range(n_base):
            J, b = add_cc(J, i, b)

            n_cf = support.n_beam_cf[i]
            if n_cf > 0:
                sl = slice(b, b + n_cf)
                c_s = curves_jax[i]
                hs = _curve_segment_hinge(
                    x_a[sl], x_b[sl], c_s.gamma(), c_s.gammadash(), dmin,
                )
                J = J + jnp.sum(jnp.where(active[sl], hs, 0.0))
                b += n_cf

        if support.stellsym:
            J, b = add_cc(J, n_base, b)

        return J

    def _J_pure(self, cdofs, sdofs):
        """Beam-to-coil free-span hinge penalty (traced scalar)."""
        curves_jax = self._curves_jax(cdofs)
        geom = self._support.beam_geometry(curves_jax, sdofs)
        x_a, x_b, active = self._effective_segments(geom)
        return self._accumulate_J(curves_jax, x_a, x_b, active)

    def _compute(self):
        """Evaluate J and its gradients from the single ``value_and_grad``."""
        if not self._needs_update:
            return
        cdofs, sdofs = self._read_dofs()
        J_val, (grad_cdofs, grad_sdofs) = self._jit_vg(cdofs, sdofs)
        self._J_cache = float(J_val)
        self._grad_curves = [np.asarray(g) for g in grad_cdofs]
        self._grad_support = grad_sdofs
        self._needs_update = False

    def J(self):
        """Beam-to-coil free-span hinge penalty (scalar)."""
        self._compute()
        return self._J_cache

    @derivative_dec
    def dJ(self):
        """Gradient of J w.r.t. the coil and support DOFs.

        Returns a :class:`~simsopt._core.derivative.Derivative` object.
        ``@derivative_dec`` contracts it into a flat numpy array aligned with
        ``self.x`` before returning to the caller.
        """
        self._compute()

        d = Derivative({})
        for curve, g in zip(self._coil_support.base_curves, self._grad_curves):
            d = d + Derivative({curve: g})
        return d + Derivative({
            self._coil_support:
                self._coil_support.flatten_grad(self._grad_support)
        })

    return_fn_map = {'J': J, 'dJ': dJ}

    def shortest_distance(self):
        """Smallest free-span-to-coil-curve distance [m].

        Returns
        -------
        float
            Minimum over active beams of the distance from segment ``S`` to
            the start coil (and, for CC beams, the end coil).  Inactive beams
            are ignored.  If every beam is inactive, returns ``inf``.
        """
        cdofs, sdofs = self._read_dofs()
        curves_jax = self._curves_jax(cdofs)
        geom = self._support.beam_geometry(curves_jax, sdofs)
        x_a, x_b, active = self._effective_segments(geom)

        support = self._support
        best = jnp.inf
        b = 0
        n_base = support.n_base

        def min_cc(g, b0, best0):
            n_g = support.n_beam_cc[g]
            if n_g == 0:
                return best0, b0
            start_idx, end_idx, end_tfm = support.cc_groups[g]
            sl = slice(b0, b0 + n_g)
            gamma_s = curves_jax[start_idx].gamma()
            gamma_e = support._apply_end_transform(
                curves_jax[end_idx].gamma(), end_tfm,
            )
            ds = jnp.min(_segment_point_dists(x_a[sl], x_b[sl], gamma_s), axis=1)
            de = jnp.min(_segment_point_dists(x_a[sl], x_b[sl], gamma_e), axis=1)
            d = jnp.minimum(ds, de)
            d = jnp.where(active[sl], d, jnp.inf)
            return jnp.minimum(best0, jnp.min(d)), b0 + n_g

        for i in range(n_base):
            best, b = min_cc(i, b, best)

            n_cf = support.n_beam_cf[i]
            if n_cf > 0:
                sl = slice(b, b + n_cf)
                ds = jnp.min(
                    _segment_point_dists(
                        x_a[sl], x_b[sl], curves_jax[i].gamma(),
                    ),
                    axis=1,
                )
                ds = jnp.where(active[sl], ds, jnp.inf)
                best = jnp.minimum(best, jnp.min(ds))
                b += n_cf

        if support.stellsym:
            best, b = min_cc(n_base, b, best)

        return float(best)


def _beam_curve_angle_hinge(abs_dot, cos_min, mask):
    """Sum of ``max(|n·t| - cos θ_min, 0)^2`` over masked beam endpoints."""
    hinge = jnp.maximum(abs_dot - cos_min, 0.0) ** 2
    return jnp.sum(jnp.where(mask, hinge, 0.0))


class BeamCurveAngle(Optimizable):
    r"""Penalise beam–coil attachments that are too nearly tangent.

    For each active attachment the hinge

    .. math::
        \max\bigl(|\mathbf{t}_\mathrm{beam}\cdot\mathbf{t}_\mathrm{coil}|
                  - \cos\theta_\min,\, 0\bigr)^2

    is summed.  Coil–coil (CC) beams contribute start and end terms; coil–
    foundation (CF) beams contribute only the start term.  ``mode`` selects
    which families enter the sum: ``'cc'``, ``'cf'``, or ``'all'``.

    Parameters
    ----------
    coil_support_beams : CoilSupportBeams
        Provides the base curves, the beam DOFs, and the underlying
        :class:`~coil_fem.coupling.SupportBeams` model.
    minimum_angle : float
        Minimum allowed beam–coil angle in radians.  Must satisfy
        ``0 <= minimum_angle < π/2``.
    mode : {'cc', 'cf', 'all'}
        Which attachment families contribute to ``J`` and
        :meth:`smallest_angle`.

    Notes
    -----
    Only the base (master) beams held in ``SupportBeams.beam_geometry`` are
    summed; stellarator-mirrored partners and field-period rotations are not
    replicated.

    Examples
    --------
    >>> Jang = BeamCurveAngle(coil_support, minimum_angle=0.2, mode='all')
    >>> Jang.smallest_angle()  # doctest: +SKIP
    0.35...
    """

    _VALID_MODES = ('cc', 'cf', 'all')

    def __init__(
        self,
        coil_support_beams,
        minimum_angle: float,
        mode: str = 'all',
    ):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for BeamCurveAngle.")

        minimum_angle = float(minimum_angle)
        if not (0.0 <= minimum_angle < 0.5 * math.pi):
            raise ValueError(
                "minimum_angle must satisfy 0 <= minimum_angle < π/2; "
                f"got {minimum_angle}."
            )
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"mode must be one of {self._VALID_MODES}; got {mode!r}."
            )

        self._coil_support = coil_support_beams
        self._support = coil_support_beams.support
        self.minimum_angle = minimum_angle
        self.mode = mode
        self._cos_min = float(math.cos(minimum_angle))

        _, beam_type = self._support.beam_labels()
        is_cc = np.asarray(beam_type == 0)
        is_cf = np.asarray(beam_type == 1)
        self._is_cc = jnp.asarray(is_cc)
        self._is_cf = jnp.asarray(is_cf)
        self._use_cc = mode in ('cc', 'all')
        self._use_cf = mode in ('cf', 'all')

        self._base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c)
            for c in coil_support_beams.base_curves
        ]

        self._jit_vg = jax.jit(value_and_grad(self._J_pure, argnums=(0, 1)))

        self._needs_update: bool = True
        self._J_cache: float | None = None
        self._grad_curves: list | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support_beams])

    def recompute_bell(self, child=None, parent=None):
        """Invalidate cached J / dJ when any ancestor DOFs change."""
        self._needs_update = True

    def _read_dofs(self):
        """Read coil / support DOFs live from the simsopt graph."""
        base_curves_dofs = [
            jnp.asarray(c.get_dofs())
            for c in self._coil_support.base_curves
        ]
        return base_curves_dofs, self._coil_support.support_dofs

    def _geom(self, cdofs, sdofs):
        """Traced ``(t_beam, t_coil_start, t_coil_end)`` for the given DOFs."""
        curves_jax = [
            CurveXYZFourierJAX(ref.quadpoints, d, ref.order)
            for ref, d in zip(self._base_curves_jax, cdofs)
        ]
        geom = self._support.beam_geometry(curves_jax, sdofs)
        return geom['t_beam'], geom['t_coil_start'], geom['t_coil_end']

    def _J_pure(self, cdofs, sdofs):
        """Beam–coil angle hinge penalty (traced scalar)."""
        t_beam, t_start, t_end = self._geom(cdofs, sdofs)
        c_s = jnp.abs(jnp.sum(t_beam * t_start, axis=-1))
        c_e = jnp.abs(jnp.sum(t_beam * t_end, axis=-1))
        cos_min = self._cos_min
        J = jnp.array(0.0)
        if self._use_cc:
            J = J + _beam_curve_angle_hinge(c_s, cos_min, self._is_cc)
            J = J + _beam_curve_angle_hinge(c_e, cos_min, self._is_cc)
        if self._use_cf:
            J = J + _beam_curve_angle_hinge(c_s, cos_min, self._is_cf)
        return J

    def _compute(self):
        """Evaluate J and its gradients from the single ``value_and_grad``."""
        if not self._needs_update:
            return
        cdofs, sdofs = self._read_dofs()
        J_val, (grad_cdofs, grad_sdofs) = self._jit_vg(cdofs, sdofs)
        self._J_cache = float(J_val)
        self._grad_curves = [np.asarray(g) for g in grad_cdofs]
        self._grad_support = grad_sdofs
        self._needs_update = False

    def J(self):
        """Beam–coil angle hinge penalty (scalar)."""
        self._compute()
        return self._J_cache

    @derivative_dec
    def dJ(self):
        """Gradient of J w.r.t. the coil and support DOFs.

        Returns a :class:`~simsopt._core.derivative.Derivative` object.
        ``@derivative_dec`` contracts it into a flat numpy array aligned with
        ``self.x`` before returning to the caller.
        """
        self._compute()

        d = Derivative({})
        for curve, g in zip(self._coil_support.base_curves, self._grad_curves):
            d = d + Derivative({curve: g})
        return d + Derivative({
            self._coil_support:
                self._coil_support.flatten_grad(self._grad_support)
        })

    return_fn_map = {'J': J, 'dJ': dJ}

    def smallest_angle(self):
        """Smallest beam–coil angle [rad] among attachments selected by ``mode``.

        Returns
        -------
        float
            ``min arccos(|t_beam · t_coil|)`` over active endpoints.  If no
            endpoints contribute (e.g. ``mode='cf'`` with no CF beams),
            returns ``π/2``.
        """
        cdofs, sdofs = self._read_dofs()
        t_beam, t_start, t_end = self._geom(cdofs, sdofs)
        c_s = jnp.abs(jnp.sum(t_beam * t_start, axis=-1))
        c_e = jnp.abs(jnp.sum(t_beam * t_end, axis=-1))

        angles = []
        if self._use_cc:
            angles.append(jnp.where(self._is_cc, jnp.arccos(jnp.clip(c_s, 0.0, 1.0)), jnp.inf))
            angles.append(jnp.where(self._is_cc, jnp.arccos(jnp.clip(c_e, 0.0, 1.0)), jnp.inf))
        if self._use_cf:
            angles.append(jnp.where(self._is_cf, jnp.arccos(jnp.clip(c_s, 0.0, 1.0)), jnp.inf))

        if not angles:
            return 0.5 * math.pi

        smallest = float(jnp.min(jnp.concatenate(angles)))
        if not np.isfinite(smallest):
            return 0.5 * math.pi
        return smallest
