"""
CoilFEM -- container class for differentiable FEM analysis of stellarator coils.

Orchestrates the full pipeline from base-coil DOFs, currents, and support
parameters to per-metric structural objectives:

    base_curves_dofs, base_currents_dofs, base_support_dofs
        -> symmetry expansion (pure JAX, differentiable)
        -> Lorentz body force at mesh quadrature points
        -> Winkler BC weights from support_fn (differentiable)
        -> JAX-FEM solve via ad_wrapper (adjoint differentiable)
        -> dict of scalar metrics (von Mises, strain energy, ...)
        -> jax.grad gives one adjoint FEM solve per base coil

Architecture choices
--------------------
- All scalar objectives are computed in a single forward pass through
  :meth:`objective`.  Using ``jax.grad`` on this function triggers exactly
  **one** adjoint FEM solve per base coil, regardless of metric count.

- Mesh topology (``cells``) and reference-element data are static (never
  traced by JAX).  Only ``points``, ``body_force``, and ``support_weights``
  flow through ``set_params``.

- Mesh points are recomputed in pure JAX by computing the curve frame at
  the stored quadrature points and broadcasting offsets -- no scipy
  interpolation, fully differentiable through DOFs.

- Body force per FEM cell is assigned by topological phi-index
  (cell_c belongs to phi slice phi_cell_indices[c]) rather than by
  physical distance, which avoids needing physical quad positions before
  the solve and keeps the assignment static.

- Winkler BC: ``support_fn(surface_points, curve_jax, dofs)`` returns per-node
  weights in ``[0, 1]`` that are absorbed into the FEM surface integral via
  ``nanson_scale``.  Gradients flow: base_support_dofs → weights → k_at_quad →
  residual → adjoint.

- For multi-GPU parallelism across independent coils, wrap the inner solve
  loop with ``jax.pmap``.  No special data structures needed.

Dependencies
------------
Requires ``jax-fem`` and ``meshio`` (both are core package dependencies).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from .geo import (
    CurveXYZFourierJAX,
    FramedCurveCentroidJAX,
    FramedCurveRMFJAX,
    apply_symmetries_to_gammas,
    apply_symmetries_to_gammadashs,
    apply_symmetries_to_currents,
    n_coils_total,
)
from .meshing import (
    rectangle_sweep,
    disk_sweep,
    _build_disk_o_grid_topology_np,
    _rect_sweep_points,
)
from .magnetic import biot_savart, B_self_quadrature
from .forces import lorentz_body_force

from jax_fem.solver import ad_wrapper
from .elasticity import (
    LinearElasticity3D,
    lame_parameters,
    recompute_fe_geometry,
)
from .metrics import (
    max_von_mises_hard,
    max_von_mises_lse,
    l2_von_mises,
    mean_von_mises_volume_weighted,
    total_strain_energy,
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

def _normalise_mesh_opts(mesh_options, n_base: int) -> list[dict]:
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


def _normalise_support_fns(base_support_fns, n_base: int) -> list[Callable]:
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


def _normalise_problem_options(problem_options: dict | None) -> dict:
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
# Pure-JAX mesh-point helpers (no CoilMesh, preserves AD chain)
# ============================================================================

def _make_framed_curve(
    curve: CurveXYZFourierJAX, frame_type: str
) -> FramedCurveRMFJAX | FramedCurveCentroidJAX:
    if frame_type == 'rmf':
        return FramedCurveRMFJAX(curve)
    elif frame_type == 'centroid':
        return FramedCurveCentroidJAX(curve)
    else:
        raise ValueError(
            f"mesh_options['frame'] must be 'rmf' or 'centroid', got {frame_type!r}."
        )


def _disk_mesh_points_jax(
    curve: CurveXYZFourierJAX,
    frame_type: str,
    radius: float,
    oxy: jax.Array,
) -> jax.Array:
    """Compute disk-sweep mesh points as a pure JAX expression.

    ``oxy`` is the static ``(n2d, 2)`` normalised O-grid offset array
    from :func:`.meshing._build_disk_o_grid_topology_np`.

    Returns ``(n_phi * n2d, 3)`` differentiable through ``curve.dofs``.
    """
    fc = _make_framed_curve(curve, frame_type)
    r0 = fc.gamma()                 # (n_phi, 3)
    _, p, q = fc.rotated_frame()  # (n_phi, 3) each

    # off shape: (n_phi, n2d, 3)
    off = radius * (
        oxy[None, :, 0:1] * p[:, None, :]
        + oxy[None, :, 1:2] * q[:, None, :]
    )
    gamma_2d = r0[:, None, :] + off   # (n_phi, n2d, 3)
    return gamma_2d.reshape(-1, 3)


# ============================================================================
# Body-force topology helpers (static, computed once at init)
# ============================================================================

def _phi_cell_indices_rect(n_phi: int, n_grid_1: int, n_grid_2: int) -> np.ndarray:
    """Return int32 array ``phi_cell_indices`` of shape ``(n_cells,)``.

    ``phi_cell_indices[c]`` is the phi (arc-length) slice index for cell ``c``
    in a rectangular-sweep TET4 mesh.

    The Freudenthal decomposition produces 6 tets per hex, and each hex belongs
    to one phi slice.  Cells are ordered as: phi first, then cross-section.

    Notes
    -----
    ``n_cells = n_phi * (n_grid_1 - 1) * (n_grid_2 - 1) * 6``
    """
    n_per_phi = (n_grid_1 - 1) * (n_grid_2 - 1) * 6
    n_cells = n_phi * n_per_phi
    return np.repeat(np.arange(n_phi, dtype=np.int32), n_per_phi)


def _phi_cell_indices_disk(n_phi: int, n_quads_2d: int) -> np.ndarray:
    """Return ``phi_cell_indices`` for a disk-sweep TET4 mesh.

    ``n_cells = n_phi * n_quads_2d * 6``
    """
    n_per_phi = n_quads_2d * 6
    return np.repeat(np.arange(n_phi, dtype=np.int32), n_per_phi)


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
        If provided, must contain ``'density'`` [kg/m³] and optionally
        ``'g_vec'`` (default ``(0, 0, -9.80665)``).
    material_options : dict or None
        Elastic and thermal material parameters.  Keys:

        * ``'E'`` : float [Pa] — Young's modulus (default 200 GPa).
        * ``'nu'`` : float — Poisson ratio (default 0.3).
        * ``'density'`` : float [kg/m³] — mass density (default 7800).
        * ``'itc'`` : float — isotropic integral thermal contraction ``ΔL/L``
          on cooldown (positive, dimensionless).  When given, the eigenstrain
          ``ε_th = −itc · I`` is pre-computed once and baked into the
          constitutive law.  ``itc`` is not a differentiable DOF.

    Notes on self-field
    -------------------
    Self-field (B_self) is always computed for every coil.  Rectangular
    cross-sections use the full Landreman-Hurwitz-Antonsen (2025) formula
    evaluated at every FEM quadrature point via
    :func:`~coil_fem.magnetic.B_self_quadrature`.  Disk cross-sections
    raise ``NotImplementedError`` (a closed-form circular analogue is known
    but not yet implemented).

    problem_options : dict or None
        Numerical solver and Winkler BC parameters.  Keys:

        * ``'winkler_k'`` : float [N/m³] — required.
        * ``'solver'`` : ``'umfpack'`` (default).
        * ``'adjoint_solver'`` : ``'umfpack'`` (default).

    Notes
    -----
    ``__init__`` builds ``LinearElasticity3D`` problems from the **initial**
    curve geometry.  Mesh topology is fixed at construction.  Subsequent calls
    pass updated ``points``, ``body_force``, and ``support_weights`` through
    ``ad_wrapper.set_params``, so the adjoint sees geometry, load, and BC
    changes without rebuilding the problem.

    Multi-GPU
    ---------
    Independent coil solves can be parallelised with ``jax.pmap``.  No
    inter-device communication is required.
    """

    """Logging verbosity level.

    * ``0`` — no logging (suppresses all JAX-FEM solver output).
    * ``1`` — INFO messages only.
    * ``2`` — DEBUG messages too (full solver verbosity).
    """

    def __init__(
        self,
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
        verbose: int = 0,
    ):
        self.verbose = verbose

        # ── 1. Validate and normalise inputs ─────────────────────────────────
        self.base_curves_jax = list(base_curves_jax)
        self.base_currents_jax = jnp.asarray(base_currents_jax, dtype=float)
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self.gravity_options = gravity_options

        n_base = len(self.base_curves_jax)
        self.mesh_opts = _normalise_mesh_opts(mesh_options, n_base)
        self.base_support_fns = _normalise_support_fns(base_support_fns, n_base)
        self._base_support_dofs = _validate_support_dofs(base_support_dofs, n_base)
        self.problem_options = _normalise_problem_options(problem_options)
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

        # ── 3. Build initial meshes; store topology + grid metadata ───────────
        self.meshes = []
        self._grid_meta = []      # list of dicts with shape-specific data

        for i, (curve, opt) in enumerate(zip(self.base_curves_jax, self.mesh_opts)):
            shape = opt['shape']
            frame_type = opt.get('frame', 'rmf')
            mesh_type = opt.get('mesh_type', 'TET4')
            fc = _make_framed_curve(curve, frame_type)
            n_phi = int(curve.quadpoints.shape[0])

            if shape == 'rect':
                init_mesh = rectangle_sweep(
                    fc, opt['w1'], opt['w2'],
                    n_grid_1=opt.get('n_grid_1'),
                    n_grid_2=opt.get('n_grid_2'),
                    aspect_ratio=opt.get('aspect_ratio', 1.0),
                    mesh_type=mesh_type,
                )
                n_cells = init_mesh.cells.shape[0]
                n_per_phi_6 = n_cells // n_phi
                n_per_phi   = n_per_phi_6 // 6
                # For TET10, n_nodes includes midpoint nodes which inflate the
                # per-slice count.  Use only the corner nodes (first 4 cols) to
                # recover the grid dimensions, which works for both TET4/TET10.
                n_corner_nodes = int(np.unique(init_mesh.cells[:, :4]).size)
                n_cross = n_corner_nodes // n_phi
                n_g1, n_g2 = _solve_grid_dims(n_cross, n_per_phi)
                # Store the metadata for the mesh. Notably, when the 
                # mesh resolution is automatically computed, the 
                # computation will only be done once in during the 
                # initialization of a CoilFEM object in the rectangle_sweep call 
                # above. After that, the mesh resolution
                # will be fixed for the rest of the optimization.
                meta = {
                    'shape': 'rect',
                    'frame': frame_type,
                    'mesh_type': mesh_type,
                    'w1': float(opt['w1']),
                    'w2': float(opt['w2']),
                    'n_grid_1': n_g1,
                    'n_grid_2': n_g2,
                    'n_phi': n_phi,
                    'n_cells': n_cells,
                    'n_quads': None,
                    'phi_cell_idx': _phi_cell_indices_rect(n_phi, n_g1, n_g2),
                    'cross_section_area': float(opt['w1']) * float(opt['w2']),
                    'phi_quad': None,   # filled after FEM problem is built
                    'uv_quad':  None,   # filled after FEM problem is built
                }

            else:  # disk
                n_center = opt.get('n_center')
                n_radial = opt.get('n_radial')
                init_mesh = disk_sweep(
                    fc, opt['radius'],
                    n_center=n_center,
                    n_radial=n_radial,
                    aspect_ratio=opt.get('aspect_ratio', 1.0),
                    mesh_type=mesh_type,
                )
                n_nodes = init_mesh.points.shape[0]
                n2d = n_nodes // n_phi
                n_cells = init_mesh.cells.shape[0]
                n_center_eff, n_radial_eff = _infer_disk_params(n2d)
                quads_np, oxy_np, _ = _build_disk_o_grid_topology_np(
                    n_center_eff, n_radial_eff
                )
                n_quads_2d = quads_np.shape[0]
                meta = {
                    'shape': 'disk',
                    'frame': frame_type,
                    'mesh_type': mesh_type,
                    'radius': float(opt['radius']),
                    'n2d': n2d,
                    'n_phi': n_phi,
                    'n_cells': n_cells,
                    'n_quads': None,
                    'phi_cell_idx': _phi_cell_indices_disk(n_phi, n_quads_2d),
                    'oxy': jnp.asarray(oxy_np, dtype=float),
                    'cross_section_area': np.pi * float(opt['radius']) ** 2,
                    'phi_quad': None,   # filled after FEM problem is built
                    'uv_quad':  None,   # disk has no (u, v) coords
                }

            self.meshes.append(init_mesh)
            self._grid_meta.append(meta)

        # ── 4. Build FEM problems and ad_wrappers (one per coil) ──────────────
        # Body force is a zero placeholder; set_params overwrites it each call.
        grav_vec = np.array(
            self.gravity_options.get('g_vec', (0.0, 0.0, -9.80665))
            if self.gravity_options else (0.0, 0.0, 0.0)
        )
        gravity_bf = (
            self._rho * grav_vec if self.gravity_options else (0.0, 0.0, 0.0)
        )

        solver_name     = self.problem_options.get('solver', 'umfpack')
        adj_solver_name = self.problem_options.get('adjoint_solver', 'umfpack')

        # cuDSS path uses its own wrapper; CPU paths use the standard ad_wrapper.
        _use_cudss = (solver_name == 'cudss')
        if not _use_cudss:
            solver_opts     = {f"{solver_name}_solver": {}}
            adj_solver_opts = {f"{adj_solver_name}_solver": {}}

        winkler_k = float(self.problem_options['winkler_k'])

        self._problems: list[LinearElasticity3D] = []
        self._fwd_preds: list = []
        # Global surface-node indices per coil — used to extract surface_pts
        # from the current mesh-points array in the forward pass.
        self._surface_node_indices: list[jnp.ndarray] = []

        thermal_info = (self._itc,)

        # Lazy import to avoid hard dependency on spineax for CPU paths.
        # cudss_solver raises an actionable ImportError if the optional GPU
        # stack (spineax + cuDSS) is missing.  The on-device assembly itself
        # lives in DeviceProblem (spineax-free) and is toggled per problem via
        # the gpu_assembly flag; only the solver wrapper needs spineax.
        if _use_cudss:
            from .backend.cudss import cudss_ad_wrapper

        for i, mesh in enumerate(self.meshes):
            # Build the FEM problem. No location_fns needed — custom_init
            # detects exterior faces topologically and builds the Winkler
            # surface structures from scratch.  gpu_assembly=True keeps the
            # Jacobian on the JAX device for the cuDSS backend.
            prob = LinearElasticity3D(
                mesh, vec=3, dim=3, ele_type=mesh.ele_type,
                additional_info=(
                    self._E, self._nu, tuple(gravity_bf), winkler_k
                ) + thermal_info,
                gpu_assembly=_use_cudss,
            )

            if self._grid_meta[i]['n_quads'] is None:
                self._grid_meta[i]['n_quads'] = len(prob.fes[0].quad_weights)

            # ── Pre-compute static reference coordinates phi_quad / uv_quad ──
            # These are coordinates (not interpolated functions), built once
            # from the mesh topology.  phi_quad values at the periodic seam
            # may exceed 1.0; interpax handles this via period=1.0.
            self._compute_ref_coords(i, prob)

            self._problems.append(prob)
            if _use_cudss:
                self._fwd_preds.append(
                    cudss_ad_wrapper(
                        prob,
                        device_id=int(self.problem_options.get('cudss_device_id', 0)),
                        mtype_id=int(self.problem_options.get('cudss_mtype_id', 1)),
                        tol=float(self.problem_options.get('cudss_tol', 1e-6)),
                        rel_tol=float(self.problem_options.get('cudss_rel_tol', 1e-8)),
                        max_iter=int(self.problem_options.get('cudss_max_iter', 50)),
                    )
                )
            else:
                self._fwd_preds.append(
                    ad_wrapper(
                        prob,
                        solver_options=solver_opts,
                        adjoint_solver_options=adj_solver_opts,
                    )
                )

            # Cache global surface node indices for this coil.
            self._surface_node_indices.append(prob.surface_node_global_indices)

    # ============================================================================
    # Static reference-coordinate pre-computation (runs once at init)
    # ============================================================================

    # VTK/JAX-FEM TET10 midpoint-to-corner edge pairs (columns 4-9).
    _TET10_MID_EDGES = np.array(
        [[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]], dtype=np.int64
    )

    def _compute_ref_coords(self, coil_idx: int, prob) -> None:
        """Pre-compute ``phi_quad`` and ``uv_quad`` for coil *coil_idx*.

        ``phi_quad[c, q]`` is the curve parameter phi at FEM quadrature point
        ``q`` of cell ``c``.  Values exceed 1.0 for cells at the periodic seam
        (last phi slice); ``interpax`` handles these via ``period=1.0``.

        ``uv_quad[c, q, :]`` holds the cross-section coordinates ``(u, v)`` in
        ``[-1, 1]`` at each FEM quadrature point (rect meshes only).

        Supports both TET4 (4-node) and TET10 (10-node) elements.  For TET10
        the reference coordinates of the 6 midpoint nodes are derived as
        averages of the corner pairs defined by the VTK/JAX-FEM edge ordering.

        For disk meshes only ``phi_quad`` is populated (``uv_quad`` stays
        ``None``).
        """
        meta  = self._grid_meta[coil_idx]
        shape = meta['shape']
        n_phi = meta['n_phi']

        fe         = prob.fes[0]
        cells_np   = np.asarray(fe.cells, dtype=np.int64)   # (n_cells, n_nodes)
        sv_np      = np.asarray(fe.shape_vals)               # (n_quads, n_nodes)
        n_cell_nodes = cells_np.shape[1]

        if n_cell_nodes not in (4, 10):
            raise ValueError(
                f"_compute_ref_coords: only TET4 and TET10 meshes are "
                f"supported; found {n_cell_nodes} nodes per element."
            )

        is_tet10 = (n_cell_nodes == 10)
        corners_np = cells_np[:, :4]  # (n_cells, 4) — corner nodes only

        phi_cell_idx_np = np.asarray(meta['phi_cell_idx'], dtype=np.int64)  # (n_cells,)

        if shape == 'rect':
            n_g1   = meta['n_grid_1']
            n_g2   = meta['n_grid_2']
            n_cross = n_g1 * n_g2
        else:  # disk
            n_cross = meta['n2d']

        # Per-corner-node phi integer (in [0, n_phi)) from global node index.
        phi_int = corners_np // n_cross          # (n_cells, 4)

        # A node is "front" (phi+1 side) if its phi-slice differs from the
        # cell's phi-slice.
        i_slice  = phi_cell_idx_np[:, np.newaxis]  # (n_cells, 1)
        is_front = (phi_int != i_slice)            # (n_cells, 4)

        # Unwrapped phi_ref: back = i/n_phi, front = (i+1)/n_phi.
        phi_corners = np.where(
            is_front,
            (i_slice + 1.0) / n_phi,
            i_slice        / n_phi,
        ).astype(np.float64)   # (n_cells, 4)

        if is_tet10:
            # Midpoint reference phi = average of the two corner endpoints.
            e = self._TET10_MID_EDGES  # (6, 2)
            phi_mids = 0.5 * (phi_corners[:, e[:, 0]] +
                              phi_corners[:, e[:, 1]])  # (n_cells, 6)
            phi_ref_local = np.concatenate(
                [phi_corners, phi_mids], axis=1
            )  # (n_cells, 10)
        else:
            phi_ref_local = phi_corners  # (n_cells, 4)

        phi_quad_np = np.einsum('qn, cn -> cq', sv_np, phi_ref_local)
        meta['phi_quad'] = jnp.asarray(phi_quad_np)   # (n_cells, n_quads)

        if shape != 'rect':
            return   # no u, v coords for disk

        # Cross-section reference coords from corner node indices.
        node_j = (corners_np % n_cross) // n_g2    # (n_cells, 4)
        node_k = corners_np % n_g2                  # (n_cells, 4)
        u_corners = (2.0 * node_j / (n_g1 - 1) - 1.0).astype(np.float64)
        v_corners = (2.0 * node_k / (n_g2 - 1) - 1.0).astype(np.float64)

        if is_tet10:
            e = self._TET10_MID_EDGES
            u_mids = 0.5 * (u_corners[:, e[:, 0]] + u_corners[:, e[:, 1]])
            v_mids = 0.5 * (v_corners[:, e[:, 0]] + v_corners[:, e[:, 1]])
            u_ref = np.concatenate([u_corners, u_mids], axis=1)  # (n_cells, 10)
            v_ref = np.concatenate([v_corners, v_mids], axis=1)
        else:
            u_ref = u_corners
            v_ref = v_corners

        uv_ref_local = np.stack([u_ref, v_ref], axis=-1)
        uv_quad_np   = np.einsum('qn, cnd -> cqd', sv_np, uv_ref_local)
        meta['uv_quad'] = jnp.asarray(uv_quad_np)

    # ============================================================================
    # Pure-JAX mesh point recomputation (preserves AD chain through dofs)
    # ============================================================================

    def _mesh_points_from_dofs(
        self, dofs_i: jax.Array, coil_idx: int
    ) -> jax.Array:
        """Compute ``(n_nodes, 3)`` mesh points as pure JAX, differentiable.

        For rectangle-sweep meshes this calls
        :func:`coil_fem.meshing._rect_sweep_points`, the same helper used at
        init-time by :func:`coil_fem.meshing.rectangle_sweep`, ensuring the
        forward-pass mesh is bit-identical to the init-time mesh.  For
        disk-sweep meshes it uses :func:`_disk_mesh_points_jax`.  The
        returned JAX array is passed as ``params['points']`` to
        ``ad_wrapper`` so the adjoint traces through the mesh geometry.
        """
        base = self.base_curves_jax[coil_idx]
        curve = CurveXYZFourierJAX(base.quadpoints, dofs_i, base.order)
        meta = self._grid_meta[coil_idx]

        if meta['shape'] == 'rect':
            fc = _make_framed_curve(curve, meta['frame'])
            pts = _rect_sweep_points(
                fc, meta['w1'], meta['w2'],
                meta['n_grid_1'], meta['n_grid_2'],
                mesh_type=meta.get('mesh_type', 'TET4'),
            )
        else:
            pts = _disk_mesh_points_jax(
                curve, meta['frame'], meta['radius'], meta['oxy']
            )
        return pts

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

        meta    = self._grid_meta[coil_idx]
        A       = meta['cross_section_area']
        n_cells = meta['n_cells']
        n_quads = meta['n_quads']
        phi_q   = meta['phi_quad']   # (n_cells, n_quads) — static

        base  = self.base_curves_jax[coil_idx]
        curve = CurveXYZFourierJAX(base.quadpoints, dofs_i, base.order)
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
        cross_section: dict = {'shape': meta['shape']}
        if meta['shape'] == 'rect':
            cross_section['w1'] = meta['w1']
            cross_section['w2'] = meta['w2']
        else:
            cross_section['radius'] = meta['radius']

        fc = _make_framed_curve(curve, meta['frame'])
        B_self_q = B_self_quadrature(
            fc, I, cross_section, phi_q, meta.get('uv_quad'),
        )   # (n_cells, n_quads, 3)

        # ── 4. B_ext at FEM quad points via Biot-Savart on physical mesh ──────
        prob_i = self._problems[coil_idx]
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
            rho = float(self.gravity_options['density'])
            g   = jnp.asarray(
                self.gravity_options.get('g_vec', (0.0, 0.0, -9.80665)),
                dtype=float,
            )
            f_vol = f_vol + (rho * g)[None, None, :]

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
        params: dict = {'points': mesh_points, 'body_force': body_force}
        if support_weights is not None:
            params['support_weights'] = support_weights
        return self._fwd_preds[coil_idx](params)

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
          directly to JAX-FEM internals (e.g. ``von_mises_stress``) that expect the
          full solution list.
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
        import contextlib
        import logging

        @contextlib.contextmanager
        def _maybe_quiet():
            jaxfem_log = logging.getLogger('jax_fem')
            old_level = jaxfem_log.level
            if self.verbose == 0:
                jaxfem_log.setLevel(logging.WARNING)
            elif self.verbose == 1:
                jaxfem_log.setLevel(logging.INFO)
            # verbose >= 2: leave level unchanged (full DEBUG output)
            try:
                yield
            finally:
                jaxfem_log.setLevel(old_level)

        from .elasticity import von_mises_stress

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
        with _maybe_quiet():
            for i in range(n_base):
                pts_i = self._mesh_points_from_dofs(base_curves_dofs[i], i)
                bf_i, B_self_i, B_ext_i = self._body_force_at_quads(
                    i, base_curves_dofs[i], pts_i,
                    all_gammas, all_gammadashs, all_currents,
                )
                weights_i = self._compute_support_weights(
                    i, pts_i, base_curves_dofs[i], sd[i]
                )
                sol = self._forward_solve(i, pts_i, bf_i, weights_i)
                vm  = von_mises_stress(self._problems[i], sol)

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
        from .elasticity import strain_tensors, recompute_fe_geometry

        result = self.run(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

        eps_total_list, eps_thermal_list = [], []
        for i in range(len(self.base_curves_jax)):
            prob = self._problems[i]
            # Recompute shape_grads from the returned geometry rather than
            # relying on the mutated prob.shape_grads state.
            sg, _, _, _ = recompute_fe_geometry(
                result['mesh_points'][i],
                prob._cells_jnp, prob._sg_ref, prob._sv, prob._qw,
            )
            eps_total, eps_thermal = strain_tensors(
                prob, result['solutions'][i], shape_grads=sg
            )
            eps_total_list.append(eps_total)
            eps_thermal_list.append(eps_thermal)

        return {
            'eps_total':   eps_total_list,    # list of (n_cells, n_quads, 3, 3)
            'eps_thermal': eps_thermal_list,  # list of (3, 3)
        }

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
            pts_i = self._mesh_points_from_dofs(base_curves_dofs[i], i)
            bf_i, _, _ = self._body_force_at_quads(
                i, base_curves_dofs[i], pts_i,
                all_gammas, all_gammadashs, all_currents,
            )
            weights_i = self._compute_support_weights(
                i, pts_i, base_curves_dofs[i], sd[i]
            )
            sol    = self._forward_solve(i, pts_i, bf_i, weights_i)
            prob_i = self._problems[i]

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
        surf_idx  = self._surface_node_indices[coil_idx]   # (n_surf_nodes,) static
        surf_pts  = pts_i[surf_idx]                         # (n_surf_nodes, 3) traced
        base      = self.base_curves_jax[coil_idx]
        coil_curr = CurveXYZFourierJAX(base.quadpoints, dofs_i, base.order)
        return self.base_support_fns[coil_idx](surf_pts, coil_curr, support_dofs_i)

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
        sd = _validate_support_dofs(base_support_dofs, n_base)

        winkler_k = float(self.problem_options['winkler_k'])

        os.makedirs(out_dir, exist_ok=True)
        written: list[str] = []

        for i, coil_mesh in enumerate(self.meshes):
            pts_i  = self._mesh_points_from_dofs(base_curves_dofs[i], i)
            pts_np = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(self._surface_node_indices[i], dtype=onp.int32)
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
            ``"viridis"``).

        Returns
        -------
        (fig, ax) : tuple
            The matplotlib figure and 3-D axes used for the plot.
        """
        import numpy as onp
        import matplotlib.pyplot as plt

        n_base = len(self.base_curves_jax)
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]
        if base_support_dofs is None:
            base_support_dofs = self._base_support_dofs
        sd = _validate_support_dofs(base_support_dofs, n_base)

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        else:
            fig = ax.figure

        sc = None
        for i in range(n_base):
            pts_i  = self._mesh_points_from_dofs(base_curves_dofs[i], i)
            pts_np = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(self._surface_node_indices[i], dtype=onp.int32)
            weights_surf = onp.asarray(
                self._compute_support_weights(i, pts_i, base_curves_dofs[i], sd[i]),
                dtype=onp.float64,
            )
            weight_full = onp.zeros(n_nodes, dtype=onp.float64)
            weight_full[surf_idx] = weights_surf

            sc = ax.scatter(
                pts_np[:, 0], pts_np[:, 1], pts_np[:, 2],
                s=s, c=weight_full, cmap=cmap, vmin=0.0, vmax=1.0,
            )

        if sc is not None:
            fig.colorbar(sc, ax=ax, label="support weight")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        return fig, ax

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
                self._surface_node_indices[i], dtype=onp.int32
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


# ============================================================================
# Geometry helpers (private utilities)
# ============================================================================

def _solve_grid_dims(n_cross: int, n_per_phi: int) -> tuple[int, int]:
    """Recover ``(n_g1, n_g2)`` from ``n_g1*n_g2 = n_cross`` and
    ``(n_g1-1)*(n_g2-1) = n_per_phi``.

    Tries all divisors of ``n_cross`` that satisfy the constraint.
    """
    for g1 in range(2, n_cross + 1):
        if n_cross % g1 == 0:
            g2 = n_cross // g1
            if (g1 - 1) * (g2 - 1) == n_per_phi:
                return g1, g2
    raise ValueError(
        f"Cannot recover grid dims: n_cross={n_cross}, n_per_phi={n_per_phi}"
    )


def _infer_disk_params(n2d: int) -> tuple[int, int]:
    """Recover ``(n_center, n_radial)`` from the total 2D node count ``n2d``
    of the O-grid disk topology.

    The O-grid formula: ``n2d = n_center^2 + 4*n_center*(n_radial-1)``
    """
    # Try all reasonable n_center values
    for nc in range(2, 50):
        remainder = n2d - nc * nc
        if remainder > 0 and remainder % (4 * nc) == 0:
            nr = remainder // (4 * nc) + 1
            if nr >= 2:
                return nc, nr
    raise ValueError(
        f"Cannot infer disk O-grid params from n2d={n2d}. "
        "Provide 'n_center' and 'n_radial' explicitly in mesh_options."
    )
