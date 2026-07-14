"""Container class for differentiable FEM structural analysis of stellarator coils.

Takes base-coil DOFs, currents, and support parameters; applies stellarator
symmetry; assembles Lorentz body forces and Winkler spring BCs; and returns
differentiable scalar structural metrics via :meth:`CoilFEM.objective`.
Calling ``jax.grad`` on :meth:`objective` triggers exactly one adjoint FEM
solve per base coil regardless of metric count.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from .geo import (
    CurveXYZFourierJAX,
    make_framed_curve,
    apply_symmetries_to_gammas,
    apply_symmetries_to_gammadashs,
    apply_symmetries_to_currents,
    n_coils_total,
)
from .meshing import CoilMesh
from .magnetic import biot_savart, B_self_quadrature, lorentz_body_force

from .problem import (
    lame_parameters,
    recompute_fe_geometry,
)
from .pipelines import ElasticPipeline, ThermoElasticPipeline
from .coupling import Support, FixedSupport
from .metrics import (
    max_von_mises_hard,
    max_von_mises_lse,
    l2_von_mises,
    mean_von_mises_volume_weighted,
    total_strain_energy,
    cauchy_stress_small_strain,
)

# ============================================================================
# Metric registry
# ============================================================================

# A registry of all metrics implemented for CoilFEM.objective.
_METRIC_REGISTRY = {
    'max_von_mises':     max_von_mises_hard,
    'max_von_mises_lse': max_von_mises_lse,
    'mean_von_mises':    mean_von_mises_volume_weighted,
    'l2_von_mises':      l2_von_mises,
    'strain_energy':     total_strain_energy,
}
# Metrics that already represent a peak (max) stress per coil.  These are
# reduced across base coils with ``max`` rather than ``sum`` in ``objective``,
# so the result is the worst-coil peak instead of a coil total. Other metrics
# are reduced by adding the per-coil values.
_METRIC_REGISTRY_MAX = frozenset({'max_von_mises', 'max_von_mises_lse'})

def _build_metric_fn(name: str):
    """Return ``(problem, sol_list, lam, mu) -> scalar`` for the given name."""
    if name not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {sorted(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[name]


# ============================================================================
# Input validation helpers
# ============================================================================

def _broadcast_mesh_opts(mesh_options, n_base: int) -> list[dict]:
    """Return a list of n_base dicts, broadcasting a single dict if needed."""
    if isinstance(mesh_options, dict):
        opts = [mesh_options] * n_base
    elif isinstance(mesh_options, (list, tuple)):
        opts = list(mesh_options)
        if len(opts) == 1:
            opts = opts * n_base
        if len(opts) != n_base:
            raise ValueError(
                f"mesh_options length ({len(opts)}) must be 1 or {n_base}."
            )
    else:
        raise TypeError(f"mesh_options must be dict or list, got {type(mesh_options)}.")

    for i, opt in enumerate(opts):
        shape = opt.get('shape')
        if shape not in ('rect', 'disk'):
            raise ValueError(
                f"mesh_options[{i}]['shape'] must be 'rect' or 'disk', got {shape!r}."
            )
        if shape == 'rect' and ('w1' not in opt or 'w2' not in opt):
            raise ValueError(
                f"mesh_options[{i}] with shape='rect' requires 'w1' and 'w2'."
            )
        if shape == 'disk' and 'radius' not in opt:
            raise ValueError(
                f"mesh_options[{i}] with shape='disk' requires 'radius'."
            )
    return opts


def _broadcast_support_fns(base_support_fns, n_base: int) -> list[Callable]:
    """Return a list of ``n_base`` support callables.

    Parameters
    ----------
    base_support_fns : callable or list[callable]
        A single ``support_fn`` (broadcast to every coil) or a list of length
        ``n_base`` with one callable per base coil.  Each callable has the
        signature ``support_fn(surface_points, curve_jax, dofs) -> weights``.

    Returns
    -------
    list[Callable] of length ``n_base``.
    """
    if callable(base_support_fns):
        return [base_support_fns] * n_base
    if isinstance(base_support_fns, (list, tuple)):
        fns = list(base_support_fns)
        if len(fns) != n_base:
            raise ValueError(
                f"base_support_fns length ({len(fns)}) must equal "
                f"n_base ({n_base})."
            )
        for i, fn in enumerate(fns):
            if not callable(fn):
                raise TypeError(
                    f"base_support_fns[{i}] must be callable, got {type(fn)}."
                )
        return fns
    raise TypeError(
        "base_support_fns must be a callable or a list of callables, "
        f"got {type(base_support_fns)}."
    )


def _validate_support_dofs(base_support_dofs, n_base: int) -> list[dict | None]:
    """Validate the shape of ``base_support_dofs``.

    Parameters
    ----------
    base_support_dofs : None or list[dict | None]
        * ``None`` — no per-coil support parameters; every coil's
          ``support_fn`` will be called with ``dofs=None``.
        * List of length ``n_base`` — element ``i`` is passed as ``dofs``
          to ``support_fn`` for coil ``i``.  Elements must be ``dict`` or
          ``None``.

    Returns
    -------
    list[dict | None] of length ``n_base``, or an empty list when
    ``base_support_dofs is None``.
    """
    if base_support_dofs is None:
        return [None] * n_base
    if not isinstance(base_support_dofs, (list, tuple)):
        raise TypeError(
            f"base_support_dofs must be None or a list, got {type(base_support_dofs)}."
        )
    if len(base_support_dofs) != n_base:
        raise ValueError(
            f"base_support_dofs length ({len(base_support_dofs)}) must equal "
            f"n_base ({n_base})."
        )
    for i, sd in enumerate(base_support_dofs):
        if sd is not None and not isinstance(sd, dict):
            raise TypeError(
                f"base_support_dofs[{i}] must be a dict or None, got {type(sd)}."
            )
    return list(base_support_dofs)


_VALID_SOLVERS = {'umfpack', 'petsc', 'jax', 'amgx', 'cudss'}


def _broadcast_problem_options(problem_options: dict | None) -> dict:
    """Validate and fill defaults for ``problem_options``.

    Parameters
    ----------
    problem_options : dict or None

    Returns
    -------
    dict with at least keys ``'winkler_k'``, ``'solver'``, ``'adjoint_solver'``.

    Recognised solver names
    -----------------------
    ``'umfpack'`` (default), ``'petsc'``, ``'jax'``, ``'amgx'``, ``'cudss'``.

    For the ``'cudss'`` GPU path, additional keys are accepted:

    * ``'cudss_device_id'`` : int, default 0 — GPU device index.
    * ``'cudss_mtype_id'``  : int, default 1 — cuDSS matrix type
      (0=general, 1=symmetric, 2=hermitian, 3=SPD, 4=HPD).
    * ``'cudss_tol'``       : float, default 1e-6 — Newton absolute tolerance.
    * ``'cudss_rel_tol'``   : float, default 1e-8 — Newton relative tolerance.
    * ``'cudss_max_iter'``  : int, default 50 — maximum Newton iterations.
    """
    opts = dict(problem_options) if problem_options else {}
    if 'winkler_k' not in opts:
        raise ValueError(
            "problem_options must contain 'winkler_k' [N/m³]."
        )
    opts.setdefault('solver', 'umfpack')
    opts.setdefault('adjoint_solver', 'umfpack')

    for key in ('solver', 'adjoint_solver'):
        val = opts[key]
        if val not in _VALID_SOLVERS:
            raise ValueError(
                f"problem_options['{key}'] = {val!r} is not recognised. "
                f"Valid choices: {sorted(_VALID_SOLVERS)}"
            )
    return opts


# ============================================================================
# CoilFEM container
# ============================================================================

class CoilFEM:
    """Differentiable FEM structural analysis container for a stellarator coil set.

    Builds the full pipeline from base-coil geometry (DOFs + currents + support
    parameters) to per-metric structural objectives.  :meth:`objective` is
    differentiable via ``jax.grad`` w.r.t. all three argument groups. Uses Winkler's 
    BC with spring constants weighted by a callable ``support_fn`` that parameterizes
    the location of support structures on each coil.

    Parameters
    ----------
    base_curves_jax : list[CurveXYZFourierJAX]
        Base coils before symmetry expansion.
    base_currents_jax : jax.Array, shape ``(n_base,)``
        Currents for the base coils [A].
    base_support_fns : callable or list[callable]
        Function(s) describing each coil's structural support via Winkler
        spring weights.  Either a single callable (broadcast to every base
        coil) or a list of length ``n_base`` with one callable per coil.
        Signature::

            support_fn(
                surface_points: jax.Array,   # (n_surface_nodes, 3)
                curve_jax: CurveXYZFourierJAX,
                dofs: dict | None,
            ) -> jax.Array                   # (n_surface_nodes,) in [0, 1]

        ``surface_points`` are the current surface-node positions (traced
        through coil DOFs).  ``dofs`` are the optimisable support parameters
        from ``base_support_dofs[i]`` for coil ``i``.  The returned weights are
        absorbed into the Winkler BC surface integral.
    base_support_dofs : list[dict | None] or None
        Per-coil initial support parameters, length ``n_base``.  Each element
        is passed as ``dofs`` to the matching ``support_fn``.  ``None`` (or a
        list of ``None`` values) means each ``support_fn`` is called with
        ``dofs=None``.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry.
    mesh_options : dict or list[dict]
        Per-coil meshing options.  Required keys:

        * ``'shape'`` : ``'rect'`` or ``'disk'``
        * ``'w1'``, ``'w2'`` (rect) or ``'radius'`` (disk)

        Optional keys:

        * ``'frame'`` : ``'rmf'`` (default) or ``'centroid'``
        * ``'mesh_type'`` : ``'TET4'`` (default)
        * ``'n_grid_1'``, ``'n_grid_2'`` (rect) or ``'n_center'``, ``'n_radial'`` (disk)
        * ``'aspect_ratio'`` : target element aspect ratio (default 1.0)

        If ``'n_grid_1'`` and ``'n_grid_2'`` are not provided, the mesh resolution
        is automatically computed based on the aspect ratio and the total length
        of the initial coil. Notably, this will only be done *once* during the 
        initialization of the CoilFEM object. After that, the mesh resolution
        will be fixed for the rest of the optimization run.

        A single dict is broadcast to all base coils.
    gravity_options : dict or None
        If provided, enables a uniform gravitational body force ``ρ·g``.  May
        contain ``'g_vec'`` (default ``(0, 0, -9.80665)``).  The mass density
        ``ρ`` is always taken from ``material_options['density']``.
    material_options : dict or None
        Elastic and thermal material parameters.  Keys:

        * ``'E'`` : float [Pa] — Young's modulus (default 200 GPa).
        * ``'nu'`` : float — Poisson ratio (default 0.3).
        * ``'density'`` : float [kg/m³] — mass density (default 7800).  Used
          both for inertial/gravity loads (when ``gravity_options`` is set)
          and reported diagnostics.
        * ``'itc'`` : float — isotropic integral thermal contraction ``ΔL/L``
          on cooldown (positive, dimensionless).  When given, the eigenstrain
          ``ε_th = −itc · I`` is pre-computed once and baked into the
          constitutive law.  ``itc`` is not a differentiable DOF.

    problem_options : dict or None
        Numerical solver and Winkler BC parameters.  Keys:

        * ``'winkler_k'`` : float [N/m³] — required.
        * ``'solver'`` : ``'umfpack'`` (default).
        * ``'adjoint_solver'`` : ``'umfpack'`` (default).

    verbose : int
        Logging verbosity for JAX-FEM output (construction and solves):

        * ``0`` (default) — no logging (suppresses all JAX-FEM solver output).
        * ``1`` — INFO messages only (``[INFO]`` lines).
        * ``2`` — DEBUG messages too (full solver verbosity).

    Notes
    -----
    **Self-field.** Self-field (B_self) is always computed for every coil.
    Rectangular cross-sections use the full Landreman-Hurwitz-Antonsen (2025)
    formula evaluated at every FEM quadrature point via
    :func:`~coil_fem.magnetic.B_self_quadrature`.  Disk cross-sections
    raise ``NotImplementedError`` (a closed-form circular analogue is known
    but not yet implemented).

    ``__init__`` builds ``LinearElasticity3D`` problems from the **initial**
    curve geometry.  Mesh topology is fixed at construction.  Subsequent calls
    pass updated ``points``, ``body_force``, and ``support_weights`` through
    ``ad_wrapper.set_params``, so the adjoint sees geometry, load, and BC
    changes without rebuilding the problem.

    ``CoilFEM`` is intentionally **not** a registered JAX pytree; it is a
    stateful container captured by closure.  Only the DOF arrays passed to
    :meth:`objective` participate in autodiff.
    """

    def __init__(
        self,
        support: Support,
        base_curves_jax: list[CurveXYZFourierJAX],
        base_currents_jax: jax.Array,
        base_support_fns: Callable | list[Callable],
        base_support_dofs: list[dict | None] | None,
        nfp: int,
        stellsym: bool,
        mesh_options: dict | list[dict],
        gravity_options: dict | None = None,
        material_options: dict | None = None,
        problem_options: dict | None = None,
        physics_options: dict | None = None,
        verbose: int = 0,
    ):
        self.verbose = verbose
        self._set_jaxfem_log_level()
        self.support = support if support is not None else FixedSupport()

        # ── 1. Validate and normalise inputs ─────────────────────────────────
        self.base_curves_jax = list(base_curves_jax)
        self.base_currents_jax = jnp.asarray(base_currents_jax, dtype=float)
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self.gravity_options = gravity_options

        n_base = len(self.base_curves_jax)
        self.mesh_opts = _broadcast_mesh_opts(mesh_options, n_base)
        self.base_support_fns = _broadcast_support_fns(base_support_fns, n_base)
        self._base_support_dofs = _validate_support_dofs(base_support_dofs, n_base)
        self.problem_options = _broadcast_problem_options(problem_options)
        self.n_total = n_coils_total(n_base, self.nfp, self.stellsym)

        # ── 2. Material properties ────────────────────────────────────────────
        mat = material_options or {}
        self._E   = float(mat.get('E', 200e9))
        self._nu  = float(mat.get('nu', 0.3))
        self._rho = float(mat.get('density', 7800.0))
        self._lam, self._mu = lame_parameters(self._E, self._nu)
        # Thermal eigenstrain parameter (optional; uniform contraction assumed).
        # ``itc`` is the positive integral thermal contraction ΔL/L applied
        # as ε_th = −itc · I.
        self._itc = float(mat['itc']) if 'itc' in mat else None

        # ── 3+4. Build per-coil pipelines (mesh + problem + fwd_pred) ───────────
        # Pipelines replace the separate self.meshes / self._problems /
        # self._fwd_preds / self._surface_node_indices lists.  The mesh is
        # built inside ElasticPipeline so topology and problem stay co-located.
        grav_vec = np.array(
            self.gravity_options.get('g_vec', (0.0, 0.0, -9.80665))
            if self.gravity_options else (0.0, 0.0, 0.0)
        )
        gravity_bf = (
            self._rho * grav_vec if self.gravity_options else (0.0, 0.0, 0.0)
        )
        winkler_k = float(self.problem_options['winkler_k'])

        _physics_type = (physics_options or {}).get('type', 'elastic')
        _valid_physics = {'elastic', 'thermoelastic'}
        if _physics_type not in _valid_physics:
            raise ValueError(
                f"physics_options['type'] = {_physics_type!r} is not recognised. "
                f"Valid choices: {sorted(_valid_physics)}"
            )

        self.pipelines: list[ElasticPipeline] = []
        for curve, opt in zip(self.base_curves_jax, self.mesh_opts):
            frame_type = opt.get('frame', 'rmf')
            mesh_type  = opt.get('mesh_type', 'TET4')
            fc   = make_framed_curve(curve, frame_type)
            mesh = CoilMesh.from_options(fc, opt, mesh_type)

            pipeline_cls = (
                ThermoElasticPipeline if _physics_type == 'thermoelastic'
                else ElasticPipeline
            )
            self.pipelines.append(
                pipeline_cls(
                    mesh, self._E, self._nu, self._itc,
                    tuple(gravity_bf), winkler_k, self.problem_options,
                )
            )

    # ============================================================================
    # Logging verbosity
    # ============================================================================

    def _set_jaxfem_log_level(self):
        """Set the ``jax_fem`` logger level from :attr:`verbose`.

        * ``verbose == 0`` — ``WARNING`` (silent).
        * ``verbose == 1`` — ``INFO``.
        * ``verbose >= 2`` — ``DEBUG``.
        """
        level = {0: logging.WARNING, 1: logging.INFO}.get(
            self.verbose, logging.DEBUG
        )
        logging.getLogger('jax_fem').setLevel(level)

    # ============================================================================
    # Symmetry expansion
    # ============================================================================

    def _expand_geometry(
        self, base_curves_dofs: list[jax.Array], base_currents_dofs: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluate base curves and expand to all symmetry images.

        Returns
        -------
        all_gammas : (n_total, n_quad, 3)
        all_gammadashs : (n_total, n_quad, 3)
        all_currents : (n_total,)
        """
        gs, gds = [], []
        for i, dofs_i in enumerate(base_curves_dofs):
            base = self.base_curves_jax[i]
            c = CurveXYZFourierJAX(base.quadpoints, dofs_i, base.order)
            gs.append(c.gamma())
            gds.append(c.gammadash())
        base_g  = jnp.stack(gs,  axis=0)
        base_gd = jnp.stack(gds, axis=0)

        all_gammas     = apply_symmetries_to_gammas(base_g, self.nfp, self.stellsym)
        all_gammadashs = apply_symmetries_to_gammadashs(base_gd, self.nfp, self.stellsym)
        all_currents   = apply_symmetries_to_currents(
            base_currents_dofs, self.nfp, self.stellsym
        )
        return all_gammas, all_gammadashs, all_currents

    # ============================================================================
    # Body-force assembly (topological phi-index assignment, no scipy)
    # ============================================================================

    def _body_force_at_quads(
        self,
        coil_idx: int,
        dofs_i: jax.Array,
        pts_i: jax.Array,
        all_gammas: jax.Array,
        all_gammadashs: jax.Array,
        all_currents: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Assemble body force and B-fields at FEM quadrature points.

        Parameters
        ----------
        coil_idx : int
        dofs_i : (n_dofs,) traced
        pts_i : (n_nodes, 3) traced — current mesh node positions, passed in
            from the caller to avoid recomputing them twice.
        all_gammas : (n_total, n_quad, 3)
        all_gammadashs : (n_total, n_quad, 3)
        all_currents : (n_total,)

        Returns
        -------
        f_vol : (n_cells, n_quads, 3) [N/m³]
            Lorentz + gravity body force at every FEM quadrature point.
        B_self_q : (n_cells, n_quads, 3) [T]
            Self-field at FEM quadrature points.
        B_ext_q : (n_cells, n_quads, 3) [T]
            Mutual field from all other coils at FEM quadrature points.

        Pipeline
        --------
        1. Interpolate tangent ``t_hat`` from centerline to FEM quad points
           via ``interpax`` periodic cubic spline.
        2. Build current density ``J_q = (I / A) * t_hat_q``.
        3. Compute ``B_self_q`` via
           :func:`~coil_fem.magnetic.B_self_quadrature` (rect; raises for disk).
        4. Compute ``B_ext_q`` via :func:`~coil_fem.magnetic.biot_savart` at
           physical quad point positions.
        5. ``f_vol = J_q × (B_self_q + B_ext_q)  +  rho * g``.
        """
        import interpax

        mesh    = self.meshes[coil_idx]
        A       = mesh.cross_section_area
        n_cells = mesh.n_cells
        n_quads = mesh.n_quads
        phi_q   = mesh.phi_quad   # (n_cells, n_quads) — static

        # Rebuild the framed curve (and its underlying curve) from the traced
        # DOFs; ``fc.curve`` is the differentiable centerline.
        fc    = mesh.framed_curve.with_dofs(dofs_i)
        curve = fc.curve
        I     = all_currents[coil_idx]

        # ── 1. Tangent at FEM quad points (interpolated, not spread) ──────────
        gammadash_cl = curve.gammadash()   # (n_phi, 3)
        t_hat_cl     = gammadash_cl / jnp.linalg.norm(
            gammadash_cl, axis=1, keepdims=True
        )
        t_hat_q = interpax.interp1d(
            phi_q.ravel(), curve.quadpoints, t_hat_cl,
            method='cubic2', period=1.0,
        ).reshape(n_cells, n_quads, 3)

        # ── 2. Current density at FEM quad points (uniform current model) ─────
        J_q = jnp.broadcast_to(
            (I / A) * t_hat_q[:, :, :],   # ensure concrete shape
            (n_cells, n_quads, 3),
        )

        # ── 3. B_self at FEM quad points ──────────────────────────────────────
        cross_section: dict = {'shape': mesh.shape}
        if mesh.shape == 'rect':
            cross_section['w1'] = mesh.w1
            cross_section['w2'] = mesh.w2
        else:
            cross_section['radius'] = mesh.radius

        B_self_q = B_self_quadrature(
            fc, I, cross_section, phi_q, mesh.uv_quad,
        )   # (n_cells, n_quads, 3)

        # ── 4. B_ext at FEM quad points via Biot-Savart on physical mesh ──────
        prob_i = self.pipelines[coil_idx].problem
        _, _, _, pqp = recompute_fe_geometry(
            pts_i, prob_i._cells_jnp, prob_i._sg_ref, prob_i._sv, prob_i._qw,
        )
        B_ext_q = biot_savart(
            pqp.reshape(-1, 3),
            all_gammas,
            all_gammadashs,
            all_currents.at[coil_idx].set(0.0),
        ).reshape(n_cells, n_quads, 3)

        # ── 5. Lorentz body force (+ gravity) ─────────────────────────────────
        f_vol = lorentz_body_force(J_q, B_self_q + B_ext_q)

        if self.gravity_options is not None:
            g = jnp.asarray(
                self.gravity_options.get('g_vec', (0.0, 0.0, -9.80665)),
                dtype=float,
            )
            f_vol = f_vol + (self._rho * g)[None, None, :]

        return f_vol, B_self_q, B_ext_q

    # ============================================================================
    # Forward solve (one per coil)
    # ============================================================================

    def _forward_solve(
        self,
        coil_idx: int,
        mesh_points: jax.Array,
        body_force: jax.Array,
        support_weights: jax.Array | None = None,
    ) -> list:
        """Run one adjoint-compatible forward FEM solve.

        Passes ``points``, ``body_force``, and (optionally) ``support_weights``
        through ``ad_wrapper``'s ``set_params`` so the adjoint traces through
        mesh geometry, loading, and Winkler BC stiffness.

        Parameters
        ----------
        mesh_points : (n_nodes, 3)
        body_force  : (n_cells, n_quads, 3)
        support_weights : (n_surface_nodes,) or None
            Per-surface-node Winkler weights in ``[0, 1]``.  Required when
            ``support_fn`` was set at construction.

        Returns
        -------
        sol_list : list[jnp.ndarray]
            Raw ``ad_wrapper`` output following JAX-FEM's multi-physics
            convention.  ``sol_list[0]`` is the displacement field, shape
            ``(n_nodes, 3)``.  For this single-physics problem the list
            always has exactly one element.
        """
        return self.pipelines[coil_idx].solve(
            mesh_points, body_force, support_weights
        )['sol_list']

    # ============================================================================
    # Public API
    # ============================================================================

    def run(
        self,
        base_curves_dofs: list[jax.Array] | None = None,
        base_currents_dofs: jax.Array | None = None,
        base_support_dofs: list[dict | None] | None = None,
    ) -> dict:
        """Forward FEM for all base coils; returns full solution dict.

        Intended for diagnostics, post-processing, and visualisation.  Does
        not compute gradients; use :meth:`objective` for optimisation.

        Logging is controlled by the :attr:`verbose` class attribute.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array] or None
            DOF vectors per base coil.  ``None`` uses initial DOFs from
            ``self.base_curves_jax``.
        base_currents_dofs : jax.Array or None
            Currents per base coil.  ``None`` uses ``self.base_currents_jax``.
        base_support_dofs : list[dict | None] or None
            Per-coil support parameters for the support functions.
            ``None`` passes ``dofs=None`` to each coil.

        Returns
        -------
        dict with keys:

        * ``'solutions'``     -- list of raw ``ad_wrapper`` outputs, one per base
          coil.  Each element is a ``list[jnp.ndarray]`` in JAX-FEM's multi-physics
          convention; ``solutions[i][0]`` has shape ``(n_nodes, 3)``.  Pass this
          directly to post-processing helpers (e.g.
          ``LinearElasticity3D.von_mises_stress``) that expect the full solution
          list.
        * ``'displacements'`` -- list of displacement arrays, one per base coil,
          shape ``(n_nodes, 3)``.  Equivalent to ``solutions[i][0]`` for each ``i``
          but exposed as a plain array for convenient post-processing.  Shares the
          same device buffer as the corresponding ``solutions`` entry (no copy).
        * ``'von_mises'``     -- list of ``(n_cells, n_quads)`` von Mises arrays
          from the combined (thermal + Lorentz + gravity) solution.
        * ``'mesh_points'``   -- list of updated ``(n_nodes, 3)`` node arrays.
        * ``'support_weights'`` -- list of ``(n_surface_nodes,)`` Winkler weight
          arrays per coil.
        * ``'f_vol'``         -- list of ``(n_cells, n_quads, 3)`` body force
          density arrays [N/m^3] per coil.
        * ``'B_self'``        -- list of ``(n_cells, n_quads, 3)`` self-field
          arrays [T] at FEM quadrature points per coil.
        * ``'B_ext'``         -- list of ``(n_cells, n_quads, 3)`` mutual
          (external) field arrays [T] at FEM quadrature points per coil.
        """
        n_base = len(self.base_curves_jax)
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]
        if base_currents_dofs is None:
            base_currents_dofs = self.base_currents_jax
        sd = _validate_support_dofs(base_support_dofs, n_base)

        all_gammas, all_gammadashs, all_currents = self._expand_geometry(
            base_curves_dofs, base_currents_dofs
        )

        sol_list, vm_list, pts_list, wt_list = [], [], [], []
        fvol_list, Bself_list, Bext_list = [], [], []
        for i in range(n_base):
            pts_i = self.meshes[i].mesh_points_from_dofs(base_curves_dofs[i])
            bf_i, B_self_i, B_ext_i = self._body_force_at_quads(
                i, base_curves_dofs[i], pts_i,
                all_gammas, all_gammadashs, all_currents,
            )
            weights_i = self._compute_support_weights(
                i, pts_i, base_curves_dofs[i], sd[i]
            )
            sol = self._forward_solve(i, pts_i, bf_i, weights_i)
            vm  = self.pipelines[i].problem.von_mises_stress(sol)

            sol_list.append(sol)
            vm_list.append(vm)
            pts_list.append(pts_i)
            wt_list.append(weights_i)
            fvol_list.append(bf_i)
            Bself_list.append(B_self_i)
            Bext_list.append(B_ext_i)

        return {
            'solutions':       sol_list,                     # list[list[array(n_nodes, 3)]]
            'displacements':   [sol[0] for sol in sol_list], # list[array(n_nodes, 3)]
            'von_mises':       vm_list,
            'mesh_points':     pts_list,
            'support_weights': wt_list,
            'f_vol':           fvol_list,   # list of (n_cells, n_quads, 3) [N/m^3]
            'B_self':          Bself_list,  # list of (n_phi, 3) [T]
            'B_ext':           Bext_list,   # list of (n_phi, 3) [T]
        }

    def compute_strain_tensors(
        self,
        base_curves_dofs: list[jax.Array] | None = None,
        base_currents_dofs: jax.Array | None = None,
        base_support_dofs: list[dict | None] | None = None,
    ) -> dict:
        """Total and thermal strain tensors per base coil.

        Runs the forward FEM (via :meth:`run`) and post-processes the
        displacement field into per-quadrature-point strain tensors.  Uses the
        small-strain additive split ``ε = ε_elastic + ε_th``: the total strain
        ``ε = ½(∇u + ∇uᵀ)`` is purely geometric, while the thermal eigenstrain
        ``ε_th = −itc · I`` is the spatially-uniform constant
        configured via ``material_options``.  The stress-producing elastic
        strain is ``eps_total - eps_thermal`` (broadcasts automatically).

        Intended for diagnostics and post-processing; no gradients are
        computed.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array] or None
            DOF vectors per base coil.  ``None`` uses initial DOFs from
            ``self.base_curves_jax``.
        base_currents_dofs : jax.Array or None
            Currents per base coil.  ``None`` uses ``self.base_currents_jax``.
        base_support_dofs : list[dict | None] or None
            Per-coil support parameters for the support functions.
            ``None`` passes ``dofs=None`` to each coil.

        Returns
        -------
        dict with keys:

        * ``'eps_total'``   -- list of ``(n_cells, n_quads, 3, 3)`` total-strain
          arrays, one per base coil.
        * ``'eps_thermal'`` -- list of ``(3, 3)`` thermal-eigenstrain arrays,
          one per base coil.  Uniform per coil (zeros when no thermal
          parameters were configured); left un-broadcast for memory efficiency.
        """
        from .problem import recompute_fe_geometry

        result = self.run(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

        eps_total_list, eps_thermal_list = [], []
        for i in range(len(self.base_curves_jax)):
            prob = self.pipelines[i].problem
            # Recompute shape_grads from the returned geometry rather than
            # relying on the mutated prob.shape_grads state.
            sg, _, _, _ = recompute_fe_geometry(
                result['mesh_points'][i],
                prob._cells_jnp, prob._sg_ref, prob._sv, prob._qw,
            )
            eps_total, eps_thermal = prob.strain_tensors(
                result['solutions'][i], shape_grads=sg
            )
            eps_total_list.append(eps_total)
            eps_thermal_list.append(eps_thermal)

        return {
            'eps_total':   eps_total_list,    # list of (n_cells, n_quads, 3, 3)
            'eps_thermal': eps_thermal_list,  # list of (3, 3)
        }

    @staticmethod
    def strain_energy_density(
        u_grad: jnp.ndarray, lam: float, mu: float, *, epsilon_th=None
    ) -> jnp.ndarray:
        """0.5 σ : ε_m — elastic (mechanical) strain-energy density per quad point.

        Uses the mechanical strain ``ε_m = ε − ε_th`` when ``epsilon_th`` is
        provided, so thermal pre-strain does not spuriously contribute to the
        elastic energy.  Consumed by the ``strain_energy`` metric
        (:func:`coil_fem.metrics.total_strain_energy`).

        Parameters
        ----------
        u_grad : jnp.ndarray, shape ``(..., 3, 3)``
            Displacement gradient at each quadrature point.
        lam, mu : float
            Lamé parameters.
        epsilon_th : jnp.ndarray or None
            Constant thermal eigenstrain ``(3, 3)``; ``None`` for isothermal.

        Returns
        -------
        jnp.ndarray, shape ``(...,)``
        """
        eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
        eps_m = eps - epsilon_th if epsilon_th is not None else eps
        sig = cauchy_stress_small_strain(u_grad, lam, mu, epsilon_th=epsilon_th)
        return 0.5 * jnp.sum(sig * eps_m, axis=(-2, -1))

    def objective(
        self,
        base_curves_dofs: list[jax.Array],
        base_currents_dofs: jax.Array,
        base_support_dofs: list[dict | None] | None = None,
        *,
        metrics: tuple[str, ...] = ('max_von_mises',),
    ) -> dict[str, jax.Array]:
        """Per-metric structural objectives, differentiable via ``jax.grad``.

        All requested metrics for a coil are accumulated inside a single
        traced expression, so ``jax.grad(self.objective)`` triggers exactly
        **one** adjoint FEM solve per base coil regardless of metric count.

        Gradients flow through all three positional arguments:

        * ``base_curves_dofs``  → coil shape via mesh geometry + Lorentz force
        * ``base_currents_dofs`` → Lorentz force
        * ``base_support_dofs`` → Winkler BC weights

        Parameters
        ----------
        base_curves_dofs : list[jax.Array]
            DOF vectors, length ``n_base``.  Each element shape
            ``(n_dofs_i,)``.
        base_currents_dofs : jax.Array, shape ``(n_base,)``
            Coil currents [A].
        base_support_dofs : list[dict | None] or None
            Per-coil support parameters passed to the support functions as
            ``dofs``.  ``None`` (default) passes ``dofs=None`` to each coil.
        metrics : tuple[str, ...]
            Metric names (static — not traced by JAX).  Available options:
            ``'max_von_mises'``, ``'max_von_mises_lse'``,
            ``'mean_von_mises'``, ``'l2_von_mises'``, ``'strain_energy'``.

        Returns
        -------
        dict[str, jax.Array]
            ``{metric_name: scalar}`` — one entry per requested metric,
            reduced over all base coils.  Max-type metrics
            (``'max_von_mises'``, ``'max_von_mises_lse'``) are reduced with
            ``max`` (worst-coil peak); all other metrics are summed.  Callers
            may weight and combine entries freely (supports augmented
            Lagrangian, Pareto, etc.).

        Examples
        --------
        Scalar objective for L-BFGS-B::

            def J(dofs, currents, support):
                objs = fem.objective(dofs, currents, support,
                                     metrics=('max_von_mises_lse',))
                return objs['max_von_mises_lse']

            grad_J = jax.grad(J, argnums=(0, 1, 2))
        """
        n_base = len(self.base_curves_jax)

        # ── Argument validation ───────────────────────────────────────────────
        if not isinstance(base_curves_dofs, (list, tuple)):
            raise TypeError(
                "base_curves_dofs must be a list of jax.Array, "
                f"got {type(base_curves_dofs)}."
            )
        if len(base_curves_dofs) != n_base:
            raise ValueError(
                f"len(base_curves_dofs) = {len(base_curves_dofs)} != "
                f"n_base = {n_base}."
            )
        base_currents_dofs = jnp.asarray(base_currents_dofs)
        if base_currents_dofs.shape != (n_base,):
            raise ValueError(
                f"base_currents_dofs.shape = {base_currents_dofs.shape}, "
                f"expected ({n_base},)."
            )
        sd = _validate_support_dofs(base_support_dofs, n_base)

        metric_fns = [_build_metric_fn(m) for m in metrics]

        # ── Symmetry expansion (shared across all coils) ──────────────────────
        all_gammas, all_gammadashs, all_currents = self._expand_geometry(
            base_curves_dofs, base_currents_dofs
        )

        # ── Per-coil solve + metric accumulation ──────────────────────────────
        # Max-type metrics reduce across coils with ``max`` (worst-coil peak);
        # all other metrics accumulate with ``sum``.
        totals = {
            m: (jnp.full((), -jnp.inf) if m in _METRIC_REGISTRY_MAX
                else jnp.zeros(()))
            for m in metrics
        }
        for i in range(n_base):
            pts_i = self.meshes[i].mesh_points_from_dofs(base_curves_dofs[i])
            bf_i, _, _ = self._body_force_at_quads(
                i, base_curves_dofs[i], pts_i,
                all_gammas, all_gammadashs, all_currents,
            )
            weights_i = self._compute_support_weights(
                i, pts_i, base_curves_dofs[i], sd[i]
            )
            sol    = self._forward_solve(i, pts_i, bf_i, weights_i)
            prob_i = self.pipelines[i].problem

            # Recompute FE geometry OUTSIDE the ad_wrapper custom_vjp scope
            # so that JAX can differentiate through shape_grads and JxW via
            # standard AD.  Reading problem.shape_grads (set as a side effect
            # inside the custom_vjp forward) would leak a traced value across
            # the custom_vjp boundary and produce NaN gradients.
            sg_ext, jxw_ext, _, _ = recompute_fe_geometry(
                pts_i, prob_i._cells_jnp,
                prob_i._sg_ref, prob_i._sv, prob_i._qw,
            )

            for m, fn in zip(metrics, metric_fns):
                val_i = fn(
                    prob_i, sol, self._lam, self._mu,
                    shape_grads=sg_ext, JxW=jxw_ext,
                )
                if m in _METRIC_REGISTRY_MAX:
                    totals[m] = jnp.maximum(totals[m], val_i)
                else:
                    totals[m] = totals[m] + val_i

        return totals

    # ============================================================================
    # Winkler weight helper (called from run and objective)
    # ============================================================================

    def _compute_support_weights(
        self,
        coil_idx: int,
        pts_i: jax.Array,
        dofs_i: jax.Array,
        support_dofs_i: dict | None,
    ) -> jax.Array:
        """Compute per-surface-node Winkler weights for coil ``coil_idx``.

        Parameters
        ----------
        pts_i : (n_nodes, 3) traced
        dofs_i : (n_dofs,) traced
        support_dofs_i : dict or None
        """
        surf_idx  = self.pipelines[coil_idx].surface_node_indices  # (n_surf_nodes,) static
        surf_pts  = pts_i[surf_idx]                               # (n_surf_nodes, 3) traced
        base      = self.base_curves_jax[coil_idx]
        coil_curr = CurveXYZFourierJAX(base.quadpoints, dofs_i, base.order)
        return self.base_support_fns[coil_idx](surf_pts, coil_curr, support_dofs_i)

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def meshes(self) -> list[CoilMesh]:
        """Per-coil mesh objects (one per base coil).

        Backward-compatibility shim: delegates to ``pipeline.mesh`` so that
        all existing code using ``self.meshes[i]`` continues to work after
        the internal migration to :class:`~coil_fem.pipelines.ElasticPipeline`.
        """
        return [p.mesh for p in self.pipelines]

    @property
    def n_nodes(self) -> int:
        """The mesh nodes count."""
        return [m.points.shape[0] for m in self.meshes]

    @property
    def n_cells(self) -> int:
        """The mesh cells count."""
        return [m.cells.shape[0] for m in self.meshes]
        
    # ============================================================================
    # Visualisation
    # ============================================================================

    @staticmethod
    def _write_coil_vtu(
        path: str,
        coil_mesh,
        pts_np,
        *,
        point_data: dict | None = None,
        cell_data: dict | None = None,
    ) -> None:
        """Write a single coil mesh to *path* as a VTU file via meshio.

        Parameters
        ----------
        path : str
            Full output path (including ``.vtu`` extension).
        coil_mesh : CoilMesh
            Mesh object supplying ``cells`` and ``meshio_cell_type``.
        pts_np : np.ndarray, shape (n_nodes, 3)
            Node coordinates as a plain NumPy array.
        point_data : dict, optional
            Per-node fields passed to :class:`meshio.Mesh` ``point_data``.
        cell_data : dict, optional
            Per-cell fields passed to :class:`meshio.Mesh` ``cell_data``.
            Each value must be a list containing one array of shape
            ``(n_cells,)`` (meshio convention).
        """
        import numpy as onp
        import meshio

        cells_np = onp.asarray(coil_mesh.cells, dtype=onp.int32)
        meshio.Mesh(
            points=pts_np,
            cells=[(coil_mesh.meshio_cell_type, cells_np)],
            point_data=point_data or {},
            cell_data=cell_data or {},
        ).write(path)

    def save_support_vtu(
        self,
        out_dir: str = ".",
        *,
        prefix: str = "coil",
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: list[dict | None] | None = None,
    ) -> list[str]:
        """Export Winkler support weights and full mesh as VTU files.

        For each base coil ``i``, writes:

        * ``{out_dir}/{prefix}{i:02d}_support.vtu`` — full tetrahedral mesh with:

          - point field ``support_weight`` in ``[0, 1]``; ``1`` = fully
            supported, ``0`` = free.
          - point field ``spring_k_Npm3`` — effective Winkler spring stiffness
            ``winkler_k × support_weight`` in N/m³.

        Open in ParaView; use *Filters → Threshold* on ``support_weight`` or
        ``spring_k_Npm3`` to isolate the clamped region.

        Parameters
        ----------
        out_dir : str
            Output directory.  Created if it does not exist.
        prefix : str
            File-name prefix (default ``"coil"``).
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``self.base_curves_jax``.
        base_support_dofs : list[dict | None] or None
            Per-coil support parameters for the support functions.

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        import os
        import numpy as onp

        n_base = len(self.base_curves_jax)
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]
        if base_support_dofs is None:
            base_support_dofs = self._base_support_dofs
        sd = _validate_support_dofs(base_support_dofs, n_base)

        winkler_k = float(self.problem_options['winkler_k'])

        os.makedirs(out_dir, exist_ok=True)
        written: list[str] = []

        for i, coil_mesh in enumerate(self.meshes):
            pts_i  = self.meshes[i].mesh_points_from_dofs(base_curves_dofs[i])
            pts_np = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(self.pipelines[i].surface_node_indices, dtype=onp.int32)
            weights_surf = onp.asarray(
                self._compute_support_weights(i, pts_i, base_curves_dofs[i], sd[i]),
                dtype=onp.float64,
            )
            weight_full = onp.zeros(n_nodes, dtype=onp.float64)
            weight_full[surf_idx] = weights_surf

            mesh_path = os.path.join(out_dir, f"{prefix}{i:02d}_support.vtu")
            self._write_coil_vtu(
                mesh_path, coil_mesh, pts_np,
                point_data={
                    "support_weights":   weight_full,
                    "spring_k_Npm3":    weight_full * winkler_k,
                },
            )
            written.append(mesh_path)

        return written

    def plot_support(
        self,
        *,
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: list[dict | None] | None = None,
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
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``self.base_curves_jax``.
        base_support_dofs : list[dict | None] or None
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

        n_base = len(self.base_curves_jax)
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]
        if base_support_dofs is None:
            base_support_dofs = self._base_support_dofs
        sd = _validate_support_dofs(base_support_dofs, n_base)

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = ax.get_figure()

        sc = None
        for i in range(n_base):
            pts_i  = self.meshes[i].mesh_points_from_dofs(base_curves_dofs[i])
            pts_np = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(self.pipelines[i].surface_node_indices, dtype=onp.int32)
            weights_surf = onp.asarray(
                self._compute_support_weights(i, pts_i, base_curves_dofs[i], sd[i]),
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

    def plot(
        self,
        *,
        base_curves_dofs: list[jax.Array] | None = None,
        base_currents_dofs: jax.Array | None = None,
        base_support_dofs: list[dict | None] | None = None,
        ax=None,
        cmap: str = "viridis",
        support_color="k",
        support_s: float = 6.0,
        axis_equal: bool = True,
    ):
        """Overlay a von Mises stress surface on the Winkler support scatter.

        Renders the exterior faces of every tetrahedron as coloured
        triangles whose colour is the owning cell's quad-averaged von Mises
        stress (in Pa). 

        Only ``TET4`` / ``TET10`` meshes are supported, since the surface
        rendering relies on triangular element faces.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array] or None
            DOF vectors per base coil.  ``None`` uses initial DOFs from
            ``self.base_curves_jax``.
        base_currents_dofs : jax.Array or None
            Currents per base coil.  ``None`` uses ``self.base_currents_jax``.
        base_support_dofs : list[dict | None] or None
            Per-coil support parameters for the support functions.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None
            Existing 3-D axes to draw on.  ``None`` (default) creates a new
            figure and 3-D axes.
        cmap : str
            Matplotlib colormap name for the von Mises surface (default
            ``"viridis"``).
        support_color : color-like
            Colour of the support markers (default ``"k"``).
        support_s : float
            Marker size for the support scatter (default ``6.0``).
        axis_equal : bool
            If ``True`` (default) scale the three axes equally via
            :func:`simsopt.geo.plotting.fix_matplotlib_3d`.

        Returns
        -------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            The 3-D axes used for the plot.  The parent figure is available as
            ``ax.get_figure()``.
        """
        import numpy as onp
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # ── Mesh-type check: surface rendering needs triangular tet faces ────
        for m in self.meshes:
            if m.ele_type not in ("TET4", "TET10"):
                raise NotImplementedError(
                    "CoilFEM.plot only supports TET4/TET10 meshes; "
                    f"found ele_type={m.ele_type!r}."
                )

        # ── Support scatter (hollow circles, weight-as-alpha) ────────────────
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = ax.get_figure()
        # ax = self.plot_support(
        #     base_curves_dofs=base_curves_dofs,
        #     base_support_dofs=base_support_dofs,
        #     ax=ax,
        #     s=support_s,
        #     simple_mode=True,
        #     color=support_color,
        # )
        # fig = ax.get_figure()

        # ── Forward FEM for von Mises + node geometry ────────────────────────
        result = self.run(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

        # Local tetrahedron faces (corner nodes only; TET10-safe).
        tet_faces = onp.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=onp.int64
        )

        coil_tris: list = []
        coil_vals: list = []
        for i, mesh in enumerate(self.meshes):
            pts_np = onp.asarray(result["mesh_points"][i], dtype=onp.float64)
            vm_cell = onp.asarray(
                jnp.mean(result["von_mises"][i], axis=-1), dtype=onp.float64
            )  # (n_cells,) [Pa]

            corners = onp.asarray(mesh.cells, dtype=onp.int64)[:, :4]  # (n_cells, 4)
            n_cells = corners.shape[0]

            faces = corners[:, tet_faces].reshape(-1, 3)              # (n_cells*4, 3)
            owner = onp.repeat(onp.arange(n_cells), tet_faces.shape[0])

            # Boundary faces appear in exactly one tet (unique sorted key).
            keys = onp.sort(faces, axis=1)
            _, inv, counts = onp.unique(
                keys, axis=0, return_inverse=True, return_counts=True
            )
            inv = onp.asarray(inv).ravel()  # numpy 2.0 may return (n, 1)
            boundary = counts[inv] == 1

            bfaces = faces[boundary]           # (n_bf, 3) node indices
            bowner = owner[boundary]           # (n_bf,)
            coil_tris.append(pts_np[bfaces])   # (n_bf, 3, 3)
            coil_vals.append(vm_cell[bowner])  # (n_bf,)

        all_vals = onp.concatenate(coil_vals) if coil_vals else onp.zeros(1)
        norm = Normalize(vmin=float(all_vals.min()), vmax=float(all_vals.max()))
        colormap = plt.get_cmap(cmap)

        for tris, vals in zip(coil_tris, coil_vals):
            coll = Poly3DCollection(tris)
            coll.set_facecolor(colormap(norm(vals)))
            coll.set_edgecolor("none")
            ax.add_collection3d(coll)

        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="von Mises [Pa]")

        if axis_equal:
            try:
                from simsopt.geo.plotting import fix_matplotlib_3d
                fix_matplotlib_3d(ax)
            except ImportError:  # pragma: no cover
                pass

        return ax

    def save_run_vtu(
        self,
        out_dir: str = ".",
        *,
        prefix: str = "coil",
        base_curves_dofs: list[jax.Array] | None = None,
        base_currents_dofs: jax.Array | None = None,
        base_support_dofs: list[dict | None] | None = None,
    ) -> list[str]:
        """Run the forward FEM solve and export results for each coil as a VTU file.

        For each base coil ``i``, writes one file:

        * ``{out_dir}/{prefix}{i:02d}_run.vtu`` — tetrahedral mesh with:

          - point field ``displacement_m`` — nodal displacement ``(n_nodes, 3)`` [m].
          - cell field ``von_mises_MPa`` — quad-averaged von Mises stress from
            the combined solution ``(n_cells,)`` [MPa].
          - cell field ``f_vol_Npm3`` — quad-averaged volumetric body-force
            vector ``(n_cells, 3)`` [N/m³].
          - cell field ``f_vol_mag_Npm3`` — magnitude of the above ``(n_cells,)``
            [N/m³].
          - cell field ``B_self_T`` / ``B_self_mag_T`` — quad-averaged self-field
            vector and magnitude at FEM quadrature points ``(n_cells, 3)`` / ``(n_cells,)`` [T].
          - cell field ``B_ext_T``  / ``B_ext_mag_T``  — quad-averaged external
            (mutual) field vector and magnitude ``(n_cells, 3)`` / ``(n_cells,)`` [T].
          - point field ``support_weight`` (``[0, 1]``) and ``spring_k_Npm3``
            (N/m³) — only written when a ``support_fn`` was supplied.

        Parameters
        ----------
        out_dir : str
            Output directory.  Created if it does not exist.
        prefix : str
            File-name prefix (default ``"coil"``).
        base_curves_dofs : list[jax.Array] or None
            DOF vectors per base coil.  ``None`` uses initial DOFs from
            ``self.base_curves_jax``.
        base_currents_dofs : jax.Array or None
            Currents per base coil.  ``None`` uses ``self.base_currents_jax``.
        base_support_dofs : list[dict | None] or None
            Per-coil support parameters for the support functions.

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        import os
        import numpy as onp
        import meshio  # noqa: F401  (import side-effect: registers VTU writer)

        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]
        if base_support_dofs is None:
            base_support_dofs = self._base_support_dofs

        sd = _validate_support_dofs(base_support_dofs, len(self.base_curves_jax))

        result = self.run(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

        winkler_k = float(self.problem_options['winkler_k'])

        os.makedirs(out_dir, exist_ok=True)
        written: list[str] = []

        for i, coil_mesh in enumerate(self.meshes):
            # ── 3-D mesh VTU (displacement, von Mises, volumetric force) ──────
            pts_np = onp.asarray(result['mesh_points'][i], dtype=onp.float64)
            n_nodes = pts_np.shape[0]
            disp   = onp.asarray(result['displacements'][i], dtype=onp.float64)  # (n_nodes, 3)
            vm_mpa = onp.asarray(
                jnp.mean(result['von_mises'][i], axis=-1) / 1e6,
                dtype=onp.float64,
            )  # (n_cells,)

            # f_vol: average over quadrature points → (n_cells, 3)
            f_vol_cell = onp.asarray(
                jnp.mean(result['f_vol'][i], axis=1),
                dtype=onp.float64,
            )

            # B fields: average over quad points → (n_cells, 3)
            B_self_cell = onp.asarray(
                jnp.mean(result['B_self'][i], axis=1), dtype=onp.float64
            )
            B_ext_cell  = onp.asarray(
                jnp.mean(result['B_ext'][i],  axis=1), dtype=onp.float64
            )

            # ── Point fields (displacement + support weights) ─────────────────
            pt_data: dict = {"displacement_m": disp}

            pts_i    = jnp.asarray(pts_np)
            surf_idx = onp.asarray(
                self.pipelines[i].surface_node_indices, dtype=onp.int32
            )
            weights_surf = onp.asarray(
                self._compute_support_weights(
                    i, pts_i, base_curves_dofs[i], sd[i]
                ),
                dtype=onp.float64,
            )
            weight_full = onp.zeros(n_nodes, dtype=onp.float64)
            weight_full[surf_idx] = weights_surf
            pt_data["support_weights"] = weight_full
            pt_data["spring_k_Npm3"]  = weight_full * winkler_k

            mesh_path = os.path.join(out_dir, f"{prefix}{i:02d}_run.vtu")
            self._write_coil_vtu(
                mesh_path, coil_mesh, pts_np,
                point_data=pt_data,
                cell_data={
                    "von_mises_MPa":         [vm_mpa],
                    "f_vol_Npm3":      [f_vol_cell],
                    "B_self_T":        [B_self_cell],
                    "B_ext_T":         [B_ext_cell],
                },
            )
            written.append(mesh_path)

        return written
