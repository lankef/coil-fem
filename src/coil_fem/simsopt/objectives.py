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
:class:`CSRVolume` estimates the central support ring volume as
``w1 * w2 * L`` from the live CSR curve.
:class:`CSRCurveDistance` hinges CSR–coil centreline clearance
(coil–coil pairs are omitted).
:class:`CSRSurfaceDistance` hinges CSR–surface clearance on one
field period (half period if stellsym).
:class:`ClampInboard` hinges fixed clamps that sit radially outboard of
each coil centre.
:class:`CRBeamInboard` hinges coil-to-CSR beam starts that sit radially
outboard of each coil centre.
"""

from __future__ import annotations

from typing import Sequence
import math
import jax
from jax import value_and_grad
import numpy as np
import jax.numpy as jnp

from ..geo import CurveXYZFourierJAX, CurveRZFourierJAX
from ..problems import recompute_fe_geometry
from ..metrics import total_strain_energy
from ..coupling import SupportBeamsCSR

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
            pts = result['mesh_points'][i]
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


# ============================================================================
# CSR volume
# ============================================================================


class CSRVolume(Optimizable):
    r"""Estimate the central support ring volume as a rectangular prism sweep.

    .. math::
        J = w_1 \, w_2 \, L,
        \qquad
        L = \bigl\langle \|\gamma'(\phi)\| \bigr\rangle_{\phi}

    where :math:`w_1` and :math:`w_2` are the static CSR cross-section widths
    and :math:`L` is the full-turn length of the live CSR
    :class:`~coil_fem.geo.CurveRZFourierJAX` (uniform quadrature over
    ``[0, 1)``).

    Parameters
    ----------
    coil_support : CoilSupportBeamsCSR
        Provides the CSR curve DOFs and the underlying
        :class:`~coil_fem.coupling.SupportBeamsCSR` (for ``w1``, ``w2``, and
        the curve template).

    Notes
    -----
    The CSR FEM mesh spans only one field period; this objective still uses
    the full-turn centreline length so ``J`` is the physical ring volume.

    Examples
    --------
    >>> Jvol = CSRVolume(coil_support)  # doctest: +SKIP
    >>> Jvol.length()  # doctest: +SKIP
    6.28...
    """

    def __init__(self, coil_support):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for CSRVolume.")

        support = coil_support.support
        if not isinstance(support, SupportBeamsCSR):
            raise TypeError(
                "CSRVolume requires coil_support.support to be a "
                f"SupportBeamsCSR; got {type(support).__name__}."
            )

        self._coil_support = coil_support
        self._support = support
        self._w1 = float(support._csr_a)
        self._w2 = float(support._csr_b)
        self._tmpl = support._csr_curve_template

        self._jit_vg = jax.jit(value_and_grad(self._J_pure, argnums=0))

        self._needs_update: bool = True
        self._J_cache: float | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support])

    def recompute_bell(self, child=None, parent=None):
        """Invalidate cached J / dJ when any ancestor DOFs change."""
        self._needs_update = True

    def _csr_curve(self, sdofs):
        """Rebuild the live CSR curve from ``sdofs['csr_curve_dofs']``."""
        tmpl = self._tmpl
        return CurveRZFourierJAX(
            tmpl.quadpoints, sdofs['csr_curve_dofs'],
            tmpl.order, tmpl.nfp, tmpl.stellsym,
        )

    def _J_pure(self, sdofs):
        """Rectangular-section CSR volume estimate (traced scalar)."""
        L = jnp.mean(self._csr_curve(sdofs).incremental_arclength())
        return self._w1 * self._w2 * L

    def _compute(self):
        """Evaluate J and its support gradient from ``value_and_grad``."""
        if not self._needs_update:
            return
        sdofs = self._coil_support.support_dofs
        J_val, grad_sdofs = self._jit_vg(sdofs)
        self._J_cache = float(J_val)
        self._grad_support = grad_sdofs
        self._needs_update = False

    def J(self):
        """Estimated CSR volume [m³]."""
        self._compute()
        return self._J_cache

    @derivative_dec
    def dJ(self):
        """Gradient of J w.r.t. the support DOFs (CSR curve coefficients).

        Returns a :class:`~simsopt._core.derivative.Derivative` object.
        ``@derivative_dec`` contracts it into a flat numpy array aligned with
        ``self.x`` before returning to the caller.
        """
        self._compute()
        return Derivative({
            self._coil_support:
                self._coil_support.flatten_grad(self._grad_support)
        })

    return_fn_map = {'J': J, 'dJ': dJ}

    def length(self):
        """Full-turn CSR centreline length [m].

        Returns
        -------
        float
            ``mean(||γ'||)`` over the CSR curve quadrature.
        """
        sdofs = self._coil_support.support_dofs
        return float(jnp.mean(self._csr_curve(sdofs).incremental_arclength()))


# ============================================================================
# CSR–coil curve distance
# ============================================================================


def _csr_curve_distance_pure(
    gamma_csr, dash_csr, gammas, dashes, minimum_distance, downsample,
):
    r"""Simsopt ``cc_distance_pure`` summed over CSR–coil pairs only.

    .. math::
        J = \sum_c \frac{1}{N_{\mathrm{csr}} N_c}
            \sum_{i,j} \|\gamma'_{\mathrm{csr}}(i)\|\,\|\gamma'_c(j)\|
            \max(0,\, d_{\min} - \|\gamma_{\mathrm{csr}}(i)-\gamma_c(j)\|)^2
    """
    gamma_csr = gamma_csr[::downsample, :]
    dash_csr = dash_csr[::downsample, :]
    n_csr = gamma_csr.shape[0]
    alen_csr = jnp.linalg.norm(dash_csr, axis=1)
    J = jnp.array(0.0)
    for gamma_c, dash_c in zip(gammas, dashes):
        gamma_c = gamma_c[::downsample, :]
        dash_c = dash_c[::downsample, :]
        dists = jnp.sqrt(jnp.sum(
            (gamma_csr[:, None, :] - gamma_c[None, :, :]) ** 2, axis=2,
        ))
        alen = alen_csr[:, None] * jnp.linalg.norm(dash_c, axis=1)[None, :]
        J = J + jnp.sum(
            alen * jnp.maximum(minimum_distance - dists, 0.0) ** 2,
        ) / (n_csr * gamma_c.shape[0])
    return J


class CSRCurveDistance(Optimizable):
    r"""Penalise a CSR centreline that comes closer than ``minimum_distance``
    to any base coil.

    The hinge matches :class:`simsopt.geo.CurveCurveDistance`, but the only
    pairs are CSR–coil.  Coil–coil pairs are omitted.  Live CSR geometry is
    rebuilt from ``support_dofs['csr_curve_dofs']``; coils come from
    :attr:`~coil_fem.simsopt.CoilSupport.base_curves`.

    .. math::
        J = \sum_{c \in \mathrm{base}}
            \frac{1}{N_{\mathrm{csr}} N_c}
            \sum_{i,j}
            \|\gamma'_{\mathrm{csr}}(i)\|\,\|\gamma'_c(j)\|
            \max\bigl(0,\, d_{\min} - \|\gamma_{\mathrm{csr}}(i)-\gamma_c(j)\|\bigr)^2

    Parameters
    ----------
    coil_support : CoilSupportBeamsCSR
        Provides the CSR curve DOFs and the base coil curves.
    minimum_distance : float
        Desired minimum CSR–coil centreline clearance [m].
    downsample : int
        Quadrature stride, as in :class:`simsopt.geo.CurveCurveDistance`.

    Notes
    -----
    Only the base coils are used; stellarator-mirrored partners and
    field-period rotations are not replicated.

    Examples
    --------
    >>> Jcc = CSRCurveDistance(coil_support, minimum_distance=0.3)  # doctest: +SKIP
    >>> Jcc.shortest_distance()  # doctest: +SKIP
    0.41...
    """

    def __init__(
        self,
        coil_support,
        minimum_distance: float,
        downsample: int = 1,
    ):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for CSRCurveDistance.")

        support = coil_support.support
        if not isinstance(support, SupportBeamsCSR):
            raise TypeError(
                "CSRCurveDistance requires coil_support.support to be a "
                f"SupportBeamsCSR; got {type(support).__name__}."
            )
        minimum_distance = float(minimum_distance)
        if minimum_distance < 0.0:
            raise ValueError(
                f"minimum_distance must be >= 0; got {minimum_distance}."
            )
        downsample = int(downsample)
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1; got {downsample}.")

        self._coil_support = coil_support
        self._support = support
        self._tmpl = support._csr_curve_template
        self.minimum_distance = minimum_distance
        self.downsample = downsample

        self._base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c)
            for c in coil_support.base_curves
        ]

        self._jit_vg = jax.jit(value_and_grad(self._J_pure, argnums=(0, 1)))

        self._needs_update: bool = True
        self._J_cache: float | None = None
        self._grad_curves: list | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support])

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

    def _csr_curve(self, sdofs):
        """Rebuild the live CSR curve from ``sdofs['csr_curve_dofs']``."""
        tmpl = self._tmpl
        return CurveRZFourierJAX(
            tmpl.quadpoints, sdofs['csr_curve_dofs'],
            tmpl.order, tmpl.nfp, tmpl.stellsym,
        )

    def _curves_jax(self, cdofs):
        """Rebuild traced coil curves from live DOFs."""
        return [
            CurveXYZFourierJAX(ref.quadpoints, d, ref.order)
            for ref, d in zip(self._base_curves_jax, cdofs)
        ]

    def _J_pure(self, cdofs, sdofs):
        """CSR–coil centreline hinge penalty (traced scalar)."""
        csr = self._csr_curve(sdofs)
        curves = self._curves_jax(cdofs)
        return _csr_curve_distance_pure(
            csr.gamma(), csr.gammadash(),
            [c.gamma() for c in curves],
            [c.gammadash() for c in curves],
            self.minimum_distance,
            self.downsample,
        )

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
        """CSR–coil centreline hinge penalty (scalar)."""
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
        """Smallest CSR–coil centreline point distance [m].

        Returns
        -------
        float
            Minimum over base coils of the sampled CSR–coil point distance.
            If there are no base coils, returns ``inf``.
        """
        cdofs, sdofs = self._read_dofs()
        g_csr = np.asarray(self._csr_curve(sdofs).gamma())[::self.downsample]
        best = np.inf
        for curve in self._curves_jax(cdofs):
            g_c = np.asarray(curve.gamma())[::self.downsample]
            dists = np.linalg.norm(
                g_csr[:, None, :] - g_c[None, :, :], axis=-1,
            )
            best = min(best, float(np.min(dists)))
        return best


class CSRSurfaceDistance(Optimizable):
    r"""Penalise a CSR centreline that comes closer than ``minimum_distance``
    to a surface.

    The hinge matches :class:`simsopt.geo.CurveSurfaceDistance`, but the
    CSR is sampled only on the first field period, or the first half
    period when the CSR is stellarator-symmetric.

    .. math::
        J = \bigl\langle
            \|\gamma'_{\mathrm{csr}}(\varphi_i)\|\,\|\mathbf{n}_s(j)\|
            \max\bigl(0,\, d_{\min} - \|\gamma_{\mathrm{csr}}(\varphi_i)
            - s_j\|\bigr)^2
        \bigr\rangle_{i,j}

    Parameters
    ----------
    coil_support : CoilSupportBeamsCSR
        Provides the CSR curve DOFs.
    surface : simsopt.geo.Surface
        Target surface.  It is *not* a DOF parent.
    minimum_distance : float
        Desired minimum CSR–surface clearance [m].
    """

    def __init__(self, coil_support, surface, minimum_distance: float):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for CSRSurfaceDistance.")

        support = coil_support.support
        if not isinstance(support, SupportBeamsCSR):
            raise TypeError(
                "CSRSurfaceDistance requires coil_support.support to be a "
                f"SupportBeamsCSR; got {type(support).__name__}."
            )
        minimum_distance = float(minimum_distance)
        if minimum_distance < 0.0:
            raise ValueError(
                f"minimum_distance must be >= 0; got {minimum_distance}."
            )

        self._coil_support = coil_support
        self._support = support
        self._tmpl = support._csr_curve_template
        self.surface = surface
        self.minimum_distance = minimum_distance

        phi_max = 1.0 / float(self._tmpl.nfp) / (
            2.0 if self._tmpl.stellsym else 1.0
        )
        qp = np.asarray(self._tmpl.quadpoints)
        self._qp_sector = jnp.asarray(qp[qp < phi_max])

        self._jit_vg = jax.jit(value_and_grad(self._J_pure, argnums=0))

        self._needs_update: bool = True
        self._J_cache: float | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support])

    def recompute_bell(self, child=None, parent=None):
        """Invalidate cached J / dJ when any ancestor DOFs change."""
        self._needs_update = True

    def _csr_curve(self, sdofs):
        """Rebuild the live CSR curve from ``sdofs['csr_curve_dofs']``."""
        tmpl = self._tmpl
        return CurveRZFourierJAX(
            tmpl.quadpoints, sdofs['csr_curve_dofs'],
            tmpl.order, tmpl.nfp, tmpl.stellsym,
        )

    def _surface_arrays(self):
        """Surface quadrature points and unnormalised normals, both ``(M, 3)``."""
        return (
            jnp.asarray(self.surface.gamma().reshape((-1, 3))),
            jnp.asarray(self.surface.normal().reshape((-1, 3))),
        )

    def _J_pure(self, sdofs, gammas, ns):
        """CSR–surface hinge penalty on the fundamental-domain samples."""
        csr = self._csr_curve(sdofs)
        qp = self._qp_sector
        gammac = csr.gamma_eval(qp)
        lc = csr.gamma_eval(qp, 1)
        dists = jnp.sqrt(jnp.sum(
            (gammac[:, None, :] - gammas[None, :, :]) ** 2, axis=2,
        ))
        integralweight = (
            jnp.linalg.norm(lc, axis=1)[:, None]
            * jnp.linalg.norm(ns, axis=1)[None, :]
        )
        return jnp.mean(
            integralweight
            * jnp.maximum(self.minimum_distance - dists, 0) ** 2
        )

    def _compute(self):
        """Evaluate J and its support gradient from ``value_and_grad``."""
        if not self._needs_update:
            return
        sdofs = self._coil_support.support_dofs
        gammas, ns = self._surface_arrays()
        J_val, grad_sdofs = self._jit_vg(sdofs, gammas, ns)
        self._J_cache = float(J_val)
        self._grad_support = grad_sdofs
        self._needs_update = False

    def J(self):
        """CSR–surface hinge penalty (scalar)."""
        self._compute()
        return self._J_cache

    @derivative_dec
    def dJ(self):
        """Gradient of J w.r.t. the support DOFs."""
        self._compute()
        return Derivative({
            self._coil_support:
                self._coil_support.flatten_grad(self._grad_support)
        })

    return_fn_map = {'J': J, 'dJ': dJ}

    def shortest_distance(self):
        """Smallest sampled CSR–surface point distance [m]."""
        sdofs = self._coil_support.support_dofs
        g_csr = np.asarray(self._csr_curve(sdofs).gamma_eval(self._qp_sector))
        gammas = np.asarray(self.surface.gamma().reshape((-1, 3)))
        return float(np.min(np.linalg.norm(
            g_csr[:, None, :] - gammas[None, :, :], axis=-1,
        )))


# ============================================================================
# Inboard attachment hinge
# ============================================================================


def _inboard_hinge_pure(curves_jax, phis_per_coil):
    r"""Sum of ``max(r - r_center, 0)^2`` over coils and attachment angles.

    Parameters
    ----------
    curves_jax : sequence of CurveXYZFourierJAX
        One curve per base coil.
    phis_per_coil : sequence of jax.Array
        Attachment angles per coil; entry ``i`` has shape ``(n_attach,)``.

    Returns
    -------
    jax.Array, shape ()
        Scalar hinge value.
    """
    J = jnp.array(0.0)
    for curve, phis in zip(curves_jax, phis_per_coil):
        c = curve.curve_center()
        r_center = jnp.sqrt(c[0] ** 2 + c[1] ** 2)
        x = curve.gamma_eval(phis)
        r = jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        J = J + jnp.sum(jnp.maximum(r - r_center, 0.0) ** 2)
    return J


class _InboardPenalty(Optimizable):
    r"""Shared hinge for attachment angles that sit outboard of a coil centre.

    For each base coil :math:`i` with centre radius
    :math:`r_{\mathrm{center},i} = \sqrt{x_{c0}^2 + y_{c0}^2}` and
    attachment angles ``phis_i``,

    .. math::
        J = \sum_i \sum_j
            \max\bigl(r_{ij} - r_{\mathrm{center},i},\, 0\bigr)^2,

    where :math:`r_{ij} = \sqrt{x_{ij}^2 + y_{ij}^2}` at
    ``gamma_eval(phis_i[j])``.

    Subclasses set ``_dof_key`` to the ``support_dofs`` key that holds the
    rectangular ``(n_coils, n_attach)`` angle array.

    Parameters
    ----------
    coil_support : CoilSupport
        Provides the base curves and the attachment-angle DOFs.
    """

    _dof_key: str = ''
    _what: str = 'attachment angles'

    def __init__(self, coil_support):
        if not _HAS_SIMSOPT:
            raise ImportError(
                f"simsopt is required for {type(self).__name__}."
            )

        sdofs = coil_support.support_dofs
        key = self._dof_key
        if key not in sdofs:
            raise TypeError(
                f"{type(self).__name__} requires support_dofs[{key!r}] "
                f"({self._what}); got keys {sorted(sdofs)}."
            )
        phis = sdofs[key]
        if np.ndim(phis) != 2 or int(np.shape(phis)[0]) != coil_support.n_coils:
            raise ValueError(
                f"support_dofs[{key!r}] must have shape "
                f"(n_coils={coil_support.n_coils}, n_attach); "
                f"got {np.shape(phis)}."
            )

        self._coil_support = coil_support
        self._base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c)
            for c in coil_support.base_curves
        ]

        self._jit_vg = jax.jit(value_and_grad(self._J_pure, argnums=(0, 1)))

        self._needs_update: bool = True
        self._J_cache: float | None = None
        self._grad_curves: list | None = None
        self._grad_support: dict | None = None

        Optimizable.__init__(self, depends_on=[coil_support])

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

    def _J_pure(self, cdofs, sdofs):
        """Outboard-attachment hinge (traced scalar)."""
        return _inboard_hinge_pure(
            self._curves_jax(cdofs), list(sdofs[self._dof_key]),
        )

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
        """Outboard-attachment hinge (scalar)."""
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

    def max_overhang(self):
        """Largest ``r - r_center`` over attachments [m].

        Returns
        -------
        float
            Positive when at least one attachment is outboard of its coil
            centre; non-positive when every attachment is inboard or on the
            centre radius.
        """
        cdofs, sdofs = self._read_dofs()
        curves = self._curves_jax(cdofs)
        phis_per_coil = list(sdofs[self._dof_key])
        best = -jnp.inf
        for curve, phis in zip(curves, phis_per_coil):
            c = curve.curve_center()
            r_center = jnp.sqrt(c[0] ** 2 + c[1] ** 2)
            x = curve.gamma_eval(phis)
            r = jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
            best = jnp.maximum(best, jnp.max(r - r_center))
        return float(best)


class ClampInboard(_InboardPenalty):
    r"""Penalise fixed clamps that sit radially outboard of a coil centre.

    Reads ``support_dofs['phis']`` of shape ``(n_coils, n_clamp)``.

    Parameters
    ----------
    coil_support : CoilSupport
        Must expose fixed-clamp angles under ``support_dofs['phis']``
        (e.g. :class:`~coil_fem.simsopt.CoilSupportFixed`).

    Examples
    --------
    >>> Jclamp = ClampInboard(coil_support)  # doctest: +SKIP
    >>> Jclamp.max_overhang()  # doctest: +SKIP
    0.12...
    """

    _dof_key = 'phis'
    _what = 'fixed-clamp angles'


class CRBeamInboard(_InboardPenalty):
    r"""Penalise CR beam starts that sit radially outboard of a coil centre.

    Reads ``support_dofs['phis_start_cr']`` of shape
    ``(n_base, n_beam_cr)``.

    Parameters
    ----------
    coil_support : CoilSupportBeamsCSR
        Must expose CR start angles under ``support_dofs['phis_start_cr']``.

    Examples
    --------
    >>> Jcr = CRBeamInboard(coil_support)  # doctest: +SKIP
    >>> Jcr.max_overhang()  # doctest: +SKIP
    0.08...
    """

    _dof_key = 'phis_start_cr'
    _what = 'CR beam start angles'
