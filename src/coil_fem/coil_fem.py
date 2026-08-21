"""Container class for differentiable FEM structural analysis of stellarator coils.

Takes base-coil DOFs, currents, and support parameters; applies stellarator
symmetry; assembles Lorentz body forces and Winkler spring BCs; and returns
differentiable scalar structural metrics via :meth:`CoilFEM.objective`.
Calling ``jax.grad`` on :meth:`objective` triggers exactly one adjoint FEM
solve per base coil regardless of metric count.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import warnings

import meshio
import numpy as np
import numpy as onp
import jax
import jax.numpy as jnp

from .geo import (
    CurveXYZFourierJAX,
    make_framed_curve,
    apply_symmetries_to_gammas,
    apply_symmetries_to_gammadashs,
    apply_symmetries_to_currents,
)
from .meshing import FramedCurveMesh
from .magnetic import biot_savart, B_self_quadrature, lorentz_body_force

from .problems import (
    lame_parameters,
    recompute_fe_geometry,
)
from .pipelines import ElasticPipeline, ThermoElasticPipeline
from .coupling import (
    Support, 
    solve_uncoupled, solve_staggered, solve_monolithic,
    MonolithicStatic, make_merged_solve,
)
from .metrics import (
    max_von_mises_hard,
    max_von_mises_lse,
    l2_von_mises,
    mean_von_mises_volume_weighted,
    total_strain_energy,
    sq_max_von_mises_lse,
)

# ============================================================================
# Metric registry
# ============================================================================

# A registry of all metrics implemented for CoilFEM.objective.
_METRIC_REGISTRY = {
    'max_von_mises':     max_von_mises_hard,
    'max_von_mises_lse': max_von_mises_lse,
    'sq_max_von_mises_lse': sq_max_von_mises_lse,
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


_VALID_SOLVERS = {'umfpack', 'petsc', 'jax', 'cudss'}


def _broadcast_problem_options(problem_options: dict | None) -> dict:
    """Validate and fill defaults for ``problem_options``.

    Parameters
    ----------
    problem_options : dict or None

    Returns
    -------
    dict with at least keys ``'solver'``, ``'adjoint_solver'``, ``'remat_bs'``.

    Recognised solver names
    -----------------------
    ``'umfpack'`` (default), ``'petsc'``, ``'jax'``, ``'cudss'``.

    For the ``'cudss'`` GPU path, additional keys are accepted:

    * ``'cudss_device_id'`` : int, default 0 — GPU device index.
    * ``'cudss_mtype_id'``  : int, optional override — cuDSS matrix type
      (0=general, 1=symmetric, 3=SPD); derived automatically from
      ``problem.matrix_symmetry``; override emits ``UserWarning``.

    Other keys
    ----------
    * ``'remat_bs'`` : bool, default True — checkpoint the Biot–Savart
      per-source scan body (see :func:`~coil_fem.magnetic.biot_savart`).
    """
    opts = dict(problem_options) if problem_options else {}
    opts.setdefault('solver', 'umfpack')
    opts.setdefault('adjoint_solver', 'umfpack')
    opts.setdefault('remat_bs', True)

    for key in ('solver', 'adjoint_solver'):
        val = opts[key]
        if val not in _VALID_SOLVERS:
            raise ValueError(
                f"problem_options['{key}'] = {val!r} is not recognised. "
                f"Valid choices: {sorted(_VALID_SOLVERS)}"
            )
    return opts


_VALID_COUPLING = {'staggered', 'monolithic'}

# ============================================================================
# CoilFEM container
# ============================================================================

class CoilFEM:
    """Differentiable FEM structural analysis container for a stellarator coil set.

    Builds the full pipeline from base-coil geometry (DOFs + currents + support
    parameters) to per-metric structural objectives.  :meth:`objective` is
    differentiable via ``jax.grad`` w.r.t. all three argument groups.  Winkler
    spring BC weights are computed by the :class:`~coil_fem.coupling.Support`
    object passed as ``support``.

    Parameters
    ----------
    base_curves_jax : list[CurveXYZFourierJAX]
        Base coils before symmetry expansion.
    base_currents_jax : jax.Array, shape ``(n_base,)``
        Currents for the base coils [A].
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
        of the initial coil.  This is done once at construction; the topology is
        then fixed for the rest of the optimisation run.

        A single dict is broadcast to all base coils.
    support : Support
        Support model owning the grounded-clamp modulus (``k_clamp``) and,
        for coupled subclasses, the beam-attachment modulus
        (``k_attachment``).  Provides per-surface-node Winkler weights via
        :meth:`~coil_fem.coupling.Support.compute_weights` and combines them
        into a stiffness via :meth:`~coil_fem.coupling.Support.stiffness`.
    gravity_options : dict or None
        If provided, enables a uniform gravitational body force ``ρ·g``.  May
        contain ``'g_vec'`` (default ``(0, 0, -9.80665)``).  The mass density
        ``ρ`` is always taken from ``material_options['density']``.
    material_options : dict or None
        Elastic and thermal material parameters.  Keys:

        * ``'E'`` : float [Pa] — Young's modulus (default 200 GPa).
        * ``'nu'`` : float — Poisson ratio (default 0.3).
        * ``'density'`` : float [kg/m³] — mass density (default 7800).
        * ``'itc'`` : float — isotropic integral thermal contraction ``ΔL/L``
          on cooldown (positive, dimensionless).  Applied as the eigenstrain
          ``ε_th = −itc · I``.  Not a differentiable DOF.

    problem_options : dict or None
        Numerical solver parameters.  Keys:

        * ``'solver'`` : ``'umfpack'`` (default).
        * ``'adjoint_solver'`` : ``'umfpack'`` (default).
        * ``'remat_bs'`` : bool (default True) — checkpoint Biot–Savart
          scan body to cut reverse-mode peak memory.

    verbose : int
        Logging verbosity (0 = silent, 1 = INFO, 2 = DEBUG).

    Notes
    -----
    ``__init__`` builds ``LinearElasticity3D`` problems from the **initial**
    curve geometry.  Mesh topology is fixed at construction.

    ``CoilFEM`` is intentionally **not** a registered JAX pytree; it is a
    stateful container captured by closure.  Only the DOF arrays passed to
    :meth:`objective` participate in autodiff.
    """

    def __init__(
        self,
        base_curves_jax: list[CurveXYZFourierJAX],
        base_currents_jax: jax.Array,
        nfp: int,
        stellsym: bool,
        mesh_options: dict | list[dict],
        support: Support,
        gravity_options: dict | None = None,
        material_options: dict | None = None,
        problem_options: dict | None = None,
        physics_options: dict | None = None,
        coupling: str = 'monolithic',
        verbose: int = 0,
    ):
        self.verbose = verbose
        self._set_jaxfem_log_level()
        self.support = support

        if coupling not in _VALID_COUPLING:
            raise ValueError(
                f"coupling={coupling!r} is not recognised. "
                f"Valid choices: {sorted(_VALID_COUPLING)}"
            )
        self.coupling = coupling

        # ── 1. Validate and normalise inputs ─────────────────────────────────
        self.base_curves_jax = list(base_curves_jax)
        self.base_currents_jax = jnp.asarray(base_currents_jax, dtype=float)
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self.gravity_options = gravity_options

        n_base = len(self.base_curves_jax)
        self.mesh_opts = _broadcast_mesh_opts(mesh_options, n_base)
        self.problem_options = _broadcast_problem_options(problem_options)

        # ── 2. Material properties ────────────────────────────────────────────
        self._E   = material_options['E']
        self._nu  = material_options['nu']
        self._rho = material_options['density']
        self._lam, self._mu = lame_parameters(self._E, self._nu)
        # Thermal eigenstrain parameter (optional; uniform contraction assumed).
        # ``itc`` is the positive integral thermal contraction ΔL/L applied
        # as ε_th = −itc · I.
        self._itc = float(material_options['itc']) if 'itc' in material_options else None

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
            mesh = FramedCurveMesh.from_options(fc, opt, mesh_type)

            pipeline_cls = (
                ThermoElasticPipeline if _physics_type == 'thermoelastic'
                else ElasticPipeline
            )
            self.pipelines.append(
                pipeline_cls(
                    mesh, self._E, self._nu, self._itc,
                    tuple(gravity_bf), self.problem_options,
                )
            )

        # Coil section metadata for SupportBeams surface-exit / L_eff.
        if hasattr(self.support, 'bind_coil_meshes'):
            self.support.bind_coil_meshes(self.meshes)

        # Monolithic CSR / cuDSS bundle is built lazily on first access to
        # :attr:`monolithic_static` (see cached_property below).

        # ── Per-coil JIT body force functions ────────────────────────────────
        # Binding coil_idx statically via functools.partial lets JAX resolve
        # self.meshes[i], self.pipelines[i], and the cross-section branch at
        # trace time, turning _body_force_at_quads into a pure traced function
        # of (dofs_i, pts_i, all_gammas, all_gammadashs, all_currents).
        self._jit_body_force_fns = [
            jax.jit(functools.partial(self._body_force_at_quads, i))
            for i in range(len(self.base_curves_jax))
        ]

        # ── Per-coil JIT recompute_fe_geometry ───────────────────────────────
        # recompute_fe_geometry is called twice per coil per objective
        # evaluation (once in _body_force_at_quads, once in the metrics loop).
        # Closing over the static problem fields eliminates re-tracing when
        # called eagerly (CPU path); on the cuDSS path the outer JIT subsumes
        # this automatically.
        def _make_fe_geom_fn(cells, sg_ref, sv, qw):
            return jax.jit(lambda pts: recompute_fe_geometry(pts, cells, sg_ref, sv, qw))

        self._jit_fe_geom_fns = [
            _make_fe_geom_fn(
                p.problem._cells_jnp, p.problem._sg_ref,
                p.problem._sv, p.problem._qw,
            )
            for p in self.pipelines
        ]

    # ============================================================================
    # Static monolithic bundle
    # ============================================================================

    @functools.cached_property
    def monolithic_static(self) -> MonolithicStatic | None:
        """Pre-built monolithic pattern / solver bundle, or ``None``.

        Built on first access when ``coupling == 'monolithic'`` and the support
        is coupled; otherwise ``None``.  Deferred so constructing / loading a
        ``CoilFEM`` with ``solver='cudss'`` does not require CUDA until a
        monolithic solve (or an explicit read of this attribute) runs.
        """
        if not (self.support.is_coupled and self.coupling == 'monolithic'):
            return None
        return self.build_monolithic_static(
            self.problem_options.get('solver', 'umfpack')
        )

    def build_monolithic_static(self, solver: str) -> MonolithicStatic:
        """Pre-build all static pattern and solver data for the monolithic solve.

        Reads ``problem.I`` / ``problem.J`` directly from each pipeline (no
        probe Jacobian assembly), merges with the support K_ss pattern from
        ``support.support_pattern``, and the coupling pattern from
        ``support.coupling_pattern``.  Builds the forward and (when
        ``solver == 'cudss'``) adjoint CSR patterns and cuDSS solver handles,
        then creates the ``custom_vjp``-wrapped ``merged_solve`` via
        :func:`~coil_fem.coupling.drivers.make_merged_solve`.

        Parameters
        ----------
        solver : str
            Value of ``problem_options['solver']``.  The cuDSS-specific layer
            (``solver_K``, ``solver_KT``, ``merged_solve``) is populated only
            when ``solver == 'cudss'``; all three fields are ``None`` otherwise.

        Returns
        -------
        MonolithicStatic
        """
        n_base = len(self.base_curves_jax)

        # ── DOF layout ───────────────────────────────────────────────────────
        n_dofs_per_coil: list[int] = [
            p.problem.num_total_dofs_all_vars for p in self.pipelines
        ]
        coil_dof_offsets: list[int] = []
        offset = 0
        for nd in n_dofs_per_coil:
            coil_dof_offsets.append(offset)
            offset += nd
        support_dof_offset = offset
        n_s = self.support.n_support_dofs
        n_total_dofs = offset + n_s

        surface_node_indices_by_coil = [
            p.surface_node_indices for p in self.pipelines
        ]

        # ── Static COO I/J for each block ────────────────────────────────────
        # Coil K_cc blocks: read problem.I/J directly, no probe assembly.
        I_blocks, J_blocks = [], []
        for i, pipeline in enumerate(self.pipelines):
            I_cc = np.asarray(pipeline.problem.I, dtype=np.int32) + coil_dof_offsets[i]
            J_cc = np.asarray(pipeline.problem.J, dtype=np.int32) + coil_dof_offsets[i]
            I_blocks.append(I_cc)
            J_blocks.append(J_cc)

        # Support K_ss block: local pattern from support, shifted to global DOFs.
        I_ss_local, J_ss_local = self.support.support_pattern()
        I_ss_pat = np.asarray(I_ss_local, dtype=np.int32) + support_dof_offset
        J_ss_pat = np.asarray(J_ss_local, dtype=np.int32) + support_dof_offset
        I_blocks.append(I_ss_pat)
        J_blocks.append(J_ss_pat)

        # Coupling K_cs / K_sc: pure numpy, no tracing.
        I_cs_pat, J_cs_pat, I_sc_pat, J_sc_pat = self.support.coupling_pattern(
            coil_dof_offsets, support_dof_offset, surface_node_indices_by_coil,
        )
        has_cs = len(I_cs_pat) > 0
        has_sc = len(I_sc_pat) > 0
        if has_cs:
            I_blocks.append(np.asarray(I_cs_pat, dtype=np.int32))
            J_blocks.append(np.asarray(J_cs_pat, dtype=np.int32))
        if has_sc:
            I_blocks.append(np.asarray(I_sc_pat, dtype=np.int32))
            J_blocks.append(np.asarray(J_sc_pat, dtype=np.int32))

        I_merged = np.concatenate(I_blocks)
        J_merged = np.concatenate(J_blocks)

        # Static curve metadata for the merged_solve closure.
        curve_qps    = tuple(c.quadpoints for c in self.base_curves_jax)
        curve_orders = tuple(c.order      for c in self.base_curves_jax)

        # ── cuDSS-specific layer ──────────────────────────────────────────────
        if solver == 'cudss':
            from .solvers.cudss import (
                _import_cudss_solver,
                build_csr_pattern,
                weakest_symmetry,
                adjoint_reuses_forward_K,
                _MTYPE_ID,
            )

            # Derive merged matrix type from each block's declared symmetry.
            _sym_claims = [p.problem.matrix_symmetry for p in self.pipelines]
            _sym_claims.append(self.support.matrix_symmetry)
            merged_sym = weakest_symmetry(*_sym_claims)
            mtype_id = _MTYPE_ID[merged_sym]
            if 'cudss_mtype_id' in self.problem_options:
                override = int(self.problem_options['cudss_mtype_id'])
                if override != mtype_id:
                    warnings.warn(
                        f"cudss_mtype_id={override} overrides derived merged "
                        f"value {mtype_id} (from weakest of {_sym_claims}). "
                        "Verify this is intentional.",
                        stacklevel=2,
                    )
                mtype_id = override
            device_id = int(self.problem_options.get('cudss_device_id', 0))
            mview_id  = 0
            adjoint_reuses_K = adjoint_reuses_forward_K(merged_sym, mtype_id)

            CuDSSSolver = _import_cudss_solver()

            def _make_solver(indptr, indices):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        message='A JAX array is being set as static!',
                        category=UserWarning,
                    )
                    return CuDSSSolver(indptr, indices, device_id, mtype_id, mview_id)

            indptr, indices, coo_to_csr, _, _, nnz_csr = build_csr_pattern(
                I_merged, J_merged, n_total_dofs
            )
            solver_K = _make_solver(indptr, indices)

            # Symmetric / SPD: Kᵀ = K — skip a second cuDSS workspace.
            if adjoint_reuses_K:
                coo_to_csr_T = nnz_csr_T = solver_KT = None
            else:
                iT, jT, coo_to_csr_T, _, _, nnz_csr_T = build_csr_pattern(
                    J_merged, I_merged, n_total_dofs
                )
                solver_KT = _make_solver(iT, jT)
        else:
            indptr = indices = coo_to_csr = None
            nnz_csr = 0
            adjoint_reuses_K = True  # unused when merged_solve is None
            coo_to_csr_T = nnz_csr_T = solver_K = solver_KT = None

        static = MonolithicStatic(
            coil_dof_offsets=tuple(coil_dof_offsets),
            support_dof_offset=support_dof_offset,
            n_total_dofs=n_total_dofs,
            n_dofs_per_coil=tuple(n_dofs_per_coil),
            n_s=n_s,
            has_cs=has_cs,
            has_sc=has_sc,
            surface_node_indices_by_coil=tuple(surface_node_indices_by_coil),
            curve_qps=curve_qps,
            curve_orders=curve_orders,
            I_ss_pat=I_ss_pat,
            J_ss_pat=J_ss_pat,
            I_cs_pat=I_cs_pat if has_cs else None,
            J_cs_pat=J_cs_pat if has_cs else None,
            I_sc_pat=I_sc_pat if has_sc else None,
            J_sc_pat=J_sc_pat if has_sc else None,
            indptr=indptr,
            indices=indices,
            coo_to_csr=coo_to_csr,
            nnz_csr=nnz_csr,
            adjoint_reuses_K=adjoint_reuses_K,
            coo_to_csr_T=coo_to_csr_T,
            nnz_csr_T=nnz_csr_T,
            solver_K=solver_K,
            solver_KT=solver_KT,
            merged_solve=None,
        )
        if solver == 'cudss':
            static = dataclasses.replace(
                static, merged_solve=make_merged_solve(self.pipelines, self.support, static)
            )
        return static

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
        for c in self.curves_from_dofs(base_curves_dofs):
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
        pqp: jax.Array | None = None,
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
        pqp : (n_cells, n_quads, 3) or None
            Pre-computed physical quadrature points from an earlier call to
            :func:`~coil_fem.problems.recompute_fe_geometry`.  When provided,
            avoids a redundant geometry recompute inside this function.
            ``None`` falls back to computing from ``pts_i``.

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
        1. Evaluate tangent ``t_hat`` at FEM quad points via
           ``curve.gamma_eval(..., diff_order=1)`` (exact Fourier derivative).
        2. Build current density ``J_q = (I / A) * t_hat_q``.
        3. Compute ``B_self_q`` via
           :func:`~coil_fem.magnetic.B_self_quadrature` (rect; raises for disk).
        4. Compute ``B_ext_q`` via :func:`~coil_fem.magnetic.biot_savart` at
           physical quad point positions.
        5. ``f_vol = J_q × (B_self_q + B_ext_q)  +  rho * g``.
        """
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

        # ── 1. Tangent at FEM quad points (exact Fourier derivative) ──────────
        gammadash_q = curve.gamma_eval(phi_q, diff_order=1)  # (n_cells, n_quads, 3)
        t_hat_q = gammadash_q / jnp.linalg.norm(
            gammadash_q, axis=-1, keepdims=True
        )

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
        if pqp is None:
            prob_i = self.pipelines[coil_idx].problem
            _, _, _, pqp = recompute_fe_geometry(
                pts_i, prob_i._cells_jnp, prob_i._sg_ref, prob_i._sv, prob_i._qw,
            )
        B_ext_q = biot_savart(
            pqp.reshape(-1, 3),
            all_gammas,
            all_gammadashs,
            all_currents.at[coil_idx].set(0.0),
            remat=self.problem_options['remat_bs'],
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

    def curves_from_dofs(self, base_curves_dofs: list) -> list:
        """Build live :class:`~coil_fem.geo.CurveXYZFourierJAX` objects from DOF arrays.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array]
            Per-base-coil DOF arrays; ``base_curves_dofs[i]`` has shape
            matching ``self.base_curves_jax[i].dofs``.

        Returns
        -------
        list[CurveXYZFourierJAX]
            One curve object per base coil, with quadpoints and order taken from
            the reference curves stored at construction.
        """
        return [
            CurveXYZFourierJAX(base.quadpoints, d, base.order)
            for base, d in zip(self.base_curves_jax, base_curves_dofs)
        ]

    # ============================================================================
    # Unified coupled/uncoupled solve helper
    # ============================================================================

    def _solve_all(
        self,
        base_curves_dofs: list[jax.Array],
        all_gammas: jax.Array,
        all_gammadashs: jax.Array,
        all_currents: jax.Array,
        base_support_dofs: dict | None,
    ) -> dict:
        """Run FEM solves for all base coils, dispatching to the correct driver.

        When ``support.is_coupled`` is ``True``, delegates to
        :func:`~coil_fem.coupling.solve_staggered` or
        :func:`~coil_fem.coupling.solve_monolithic` depending on
        :attr:`coupling`.  Otherwise runs an independent per-coil loop.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array]
        all_gammas, all_gammadashs : (n_total, n_quad, 3)
        all_currents : (n_total,)
        base_support_dofs : dict or None

        Returns
        -------
        dict with keys:

        * ``'sol_list_by_coil'`` — list of ``sol_list`` (each a list of arrays),
          one per base coil.
        * ``'pts_by_coil'``     — list of ``(n_nodes, 3)`` mesh points.
        * ``'bf_by_coil'``      — list of body-force arrays.
        * ``'B_self_by_coil'``  — list of ``(n_cells, n_quads, 3)`` self-field arrays.
        * ``'B_ext_by_coil'``   — list of ``(n_cells, n_quads, 3)`` mutual-field arrays.
        * ``'stiffness_by_coil'`` — list of per-quad Winkler stiffness arrays.
        * ``'u_s'``             — ``(n_support_dofs,)`` support DOF vector, or
          ``None`` for uncoupled solves.
        """
        n_base = len(self.base_curves_jax)

        # Build live CurveXYZFourierJAX objects once (used for weights + drivers).
        curves_jax_list = self.curves_from_dofs(base_curves_dofs)

        # When coupled, compute beam geometry once and reuse for *forward*
        # weights + monolithic assemble.  This is a forward-only cache: the
        # custom_vjp constraint in make_merged_solve must recompute
        # beam_geometry from support DOFs so ∂K/∂φ reaches the adjoint.
        # Do not freeze geom in that VJP as a "memory optimization".
        support_geom = None
        if self.support.is_coupled:
            support_geom = self.support.beam_geometry(
                curves_jax_list, base_support_dofs or {}
            )

        pts_by_coil     = []
        bf_by_coil      = []
        bself_by_coil   = []
        bext_by_coil    = []
        k_by_coil       = []
        sg_by_coil      = []   # shape_grads for objective metric reuse
        jxw_by_coil_fe  = []   # JxW for objective metric reuse
        fe_geom_by_coil = []   # full (sg, jxw, vgj, pqp) for set_params reuse
        for i in range(n_base):
            pts_i = self.meshes[i].mesh_points_from_dofs(base_curves_dofs[i])
            # Compute FE geometry once per coil; thread across body force,
            # set_params, and the objective metrics loop (review item 2d).
            sg_i, jxw_i, vgj_i, pqp_i = self._jit_fe_geom_fns[i](pts_i)
            bf_i, bself_i, bext_i = self._jit_body_force_fns[i](
                base_curves_dofs[i], pts_i,
                all_gammas, all_gammadashs, all_currents,
                pqp_i,
            )
            w_g_i, w_a_i = self._support_weights(
                i, pts_i, curves_jax_list, base_support_dofs,
                geom=support_geom,
            )
            k_i = self.support.stiffness(w_g_i, w_a_i)
            pts_by_coil.append(pts_i)
            bf_by_coil.append(bf_i)
            bself_by_coil.append(bself_i)
            bext_by_coil.append(bext_i)
            k_by_coil.append(k_i)
            sg_by_coil.append(sg_i)
            jxw_by_coil_fe.append(jxw_i)
            fe_geom_by_coil.append((sg_i, jxw_i, vgj_i, pqp_i))

        driver_params = {
            'mesh_points_by_coil': pts_by_coil,
            'body_force_by_coil':  bf_by_coil,
            'stiffness_by_coil':   k_by_coil,
            'curves_by_coil':      curves_jax_list,
            'support_dofs':        base_support_dofs or {},
            # Pre-computed FE geometry (sg, jxw, vgj, pqp) per coil; passed to
            # pipeline.solve / monolithic assemble → set_params to skip
            # recompute_fe_geometry there.
            'fe_geom_by_coil':     fe_geom_by_coil,
            # Pre-computed beam geometry for the forward assemble only.
            # Adjoint path in make_merged_solve recomputes beam_geometry.
            'support_geom':        support_geom,
        }

        if self.support.is_coupled:
            if self.coupling == 'monolithic':
                result = solve_monolithic(
                    self.pipelines, self.support, driver_params,
                    self.monolithic_static,
                )
            else:
                result = solve_staggered(
                    self.pipelines, self.support, driver_params
                )
        else:
            result = solve_uncoupled(self.pipelines, self.support, driver_params)

        return {
            'sol_list_by_coil':   result['sol_list_by_coil'],
            'pts_by_coil':        pts_by_coil,
            'bf_by_coil':         bf_by_coil,
            'B_self_by_coil':     bself_by_coil,
            'B_ext_by_coil':      bext_by_coil,
            'stiffness_by_coil':  k_by_coil,
            'sg_by_coil':         sg_by_coil,
            'jxw_by_coil_fe':     jxw_by_coil_fe,
            'u_s':                result['u_s'],
            'diagnostics':        result.get('diagnostics', {}),
        }

    # ============================================================================
    # Public API
    # ============================================================================

    def run(
        self,
        base_curves_dofs: list[jax.Array] | None = None,
        base_currents_dofs: jax.Array | None = None,
        base_support_dofs: dict | None = None,
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
        base_support_dofs : dict or None
            Merged support-dofs dict for the whole coil set (as returned by
            :attr:`~coil_fem.simsopt.CoilSupport.support_dofs`).  ``None``
            lets the support object use its own default parameters.

        Returns
        -------
        dict with keys:

        * ``'solutions'``     -- list of raw ``ad_wrapper`` outputs, one per base
          coil.  Each element is a ``list[jnp.ndarray]`` in JAX-FEM's multi-physics
          convention; ``solutions[i][0]`` has shape ``(n_nodes, 3)``.
        * ``'displacements'`` -- list of displacement arrays, one per base coil,
          shape ``(n_nodes, 3)``.
        * ``'von_mises'``     -- list of ``(n_cells, n_quads)`` von Mises arrays.
        * ``'mesh_points'``   -- list of updated ``(n_nodes, 3)`` node arrays.
        * ``'support_k'``     -- list of ``(n_surface_quads,)`` Winkler stiffness
          arrays [N/m³] per coil (one entry per surface quadrature point).
        * ``'f_vol'``         -- list of ``(n_cells, n_quads, 3)`` body force
          density arrays [N/m^3] per coil.
        * ``'B_self'``        -- list of ``(n_cells, n_quads, 3)`` self-field
          arrays [T] per coil.
        * ``'B_ext'``         -- list of ``(n_cells, n_quads, 3)`` mutual field
          arrays [T] per coil.
        * ``'u_s'``           -- support DOF vector, shape ``(n_support_dofs,)``,
          or ``None`` for an uncoupled :class:`~coil_fem.coupling.Support`.
          For :class:`~coil_fem.coupling.SupportBeams`, reshape via
          ``self.support.endpoint_state(result['u_s'])`` for a per-beam,
          per-node ``(translation, rotation)`` breakdown, or evaluate
          ``self.support.beam_displacement(geom, result['u_s'], xi)`` for the
          closed-form displacement along each beam.
        """
        n_base = len(self.base_curves_jax)
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]
        if base_currents_dofs is None:
            base_currents_dofs = self.base_currents_jax

        all_gammas, all_gammadashs, all_currents = self._expand_geometry(
            base_curves_dofs, base_currents_dofs
        )

        solved = self._solve_all(
            base_curves_dofs, all_gammas, all_gammadashs, all_currents,
            base_support_dofs,
        )

        # B_self / B_ext come from the same _body_force_at_quads calls already
        # made inside _solve_all; no second Biot-Savart pass needed here.
        vm_list = [
            self.pipelines[i].problem.von_mises_stress(solved['sol_list_by_coil'][i])
            for i in range(n_base)
        ]

        sol_list = solved['sol_list_by_coil']
        return {
            'solutions':       sol_list,                     # list[list[array(n_nodes, 3)]]
            'displacements':   [sol[0] for sol in sol_list], # list[array(n_nodes, 3)]
            'von_mises':       vm_list,
            'mesh_points':     solved['pts_by_coil'],
            'support_k':       solved['stiffness_by_coil'],
            'f_vol':           solved['bf_by_coil'],         # list of (n_cells, n_quads, 3) [N/m^3]
            'B_self':          solved['B_self_by_coil'],     # list of (n_cells, n_quads, 3) [T]
            'B_ext':           solved['B_ext_by_coil'],      # list of (n_cells, n_quads, 3) [T]
            'u_s':             solved['u_s'],                # (n_support_dofs,) or None
        }

    def compute_strain_tensors(
        self,
        base_curves_dofs: list[jax.Array] | None = None,
        base_currents_dofs: jax.Array | None = None,
        base_support_dofs: dict | None = None,
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
        base_support_dofs : dict or None
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

    def objective(
        self,
        base_curves_dofs: list[jax.Array],
        base_currents_dofs: jax.Array,
        base_support_dofs: dict | None = None,
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
            DOF vectors, length ``n_base``.
        base_currents_dofs : jax.Array, shape ``(n_base,)``
            Coil currents [A].
        base_support_dofs : dict or None
            Merged support-dofs dict for the whole coil set (as returned by
            :attr:`~coil_fem.simsopt.CoilSupport.support_dofs`).  ``None``
            lets the support object use its own default parameters.
        metrics : tuple[str, ...]
            Metric names (static).  Available: ``'max_von_mises'``,
            ``'max_von_mises_lse'``, ``'mean_von_mises'``,
            ``'l2_von_mises'``, ``'strain_energy'``.

        Returns
        -------
        dict[str, jax.Array]
            ``{metric_name: scalar}`` reduced over all base coils.

        Examples
        --------
        Scalar objective for L-BFGS-B::

            def J(dofs, currents, support_dofs):
                objs = fem.objective(dofs, currents, support_dofs,
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

        metric_fns = [_build_metric_fn(m) for m in metrics]

        # ── Symmetry expansion (shared across all coils) ──────────────────────
        all_gammas, all_gammadashs, all_currents = self._expand_geometry(
            base_curves_dofs, base_currents_dofs
        )

        # ── Solve all coils (coupled or uncoupled) ────────────────────────────
        solved = self._solve_all(
            base_curves_dofs, all_gammas, all_gammadashs, all_currents,
            base_support_dofs,
        )

        # ── Per-coil metric accumulation ──────────────────────────────────────
        # Max-type metrics reduce across coils with ``max`` (worst-coil peak);
        # all other metrics accumulate with ``sum``.
        totals = {
            m: (jnp.full((), -jnp.inf) if m in _METRIC_REGISTRY_MAX
                else jnp.zeros(()))
            for m in metrics
        }
        for i in range(n_base):
            pts_i  = solved['pts_by_coil'][i]
            sol    = solved['sol_list_by_coil'][i]
            prob_i = self.pipelines[i].problem

            # Use the FE geometry pre-computed in _solve_all instead of
            # recomputing from pts_i.  This is correct because pts_i in the
            # metrics loop is the same array that was passed to
            # _jit_fe_geom_fns[i] inside _solve_all; the cached (sg, JxW) are
            # therefore consistent with sol.  Avoids a third
            # recompute_fe_geometry call per coil per objective evaluation.
            sg_ext  = solved['sg_by_coil'][i]
            jxw_ext = solved['jxw_by_coil_fe'][i]

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
    # Winkler weight helper (called from run, objective, and visualisation)
    # ============================================================================

    def _support_weights(
        self,
        coil_idx: int,
        pts_i: jax.Array,
        curves_jax: list,
        support_dofs: dict | None,
        *,
        geom: dict | None = None,
        at: str = 'quads',
    ) -> tuple[jax.Array, jax.Array]:
        """Grounded-clamp and beam-attachment weights for coil ``coil_idx``.

        Parameters
        ----------
        pts_i : (n_nodes, 3) traced
        curves_jax : list[CurveXYZFourierJAX]
            Differentiable centreline curves for **all** base coils.
        support_dofs : dict or None
            Full merged support-dofs dict for the coil set.  Passed directly to
            :meth:`~coil_fem.coupling.Support.compute_weights`.
        geom : dict or None
            Pre-computed beam geometry from
            :meth:`~coil_fem.coupling.Support.beam_geometry`.
        at : ``'quads'`` or ``'nodes'``
            Point set at which to evaluate the weight function.  ``'quads'``
            (default) returns ``(n_surface_quads,)`` arrays for the solve;
            ``'nodes'`` returns ``(n_surface_nodes,)`` arrays for visualisation.

        Returns
        -------
        w_g, w_a : jax.Array
            Grounded-clamp and beam-attachment weight fields.
        """
        pipeline = self.pipelines[coil_idx]
        if at == 'nodes':
            surf_pts = pts_i[pipeline.surface_node_indices]
        else:
            surf_pts = pipeline.surface_quad_points(pts_i)
        kw = {'geom': geom} if geom is not None else {}
        return self.support.compute_weights(
            coil_idx, surf_pts, curves_jax, support_dofs, **kw
        )

    # ============================================================================
    # Properties
    # ============================================================================

    @functools.cached_property
    def meshes(self) -> list[FramedCurveMesh]:
        """Per-coil mesh objects (one per base coil).

        Backward-compatibility shim: delegates to ``pipeline.mesh`` so that
        all existing code using ``self.meshes[i]`` continues to work after
        the internal migration to :class:`~coil_fem.pipelines.ElasticPipeline`.
        """
        return [p.mesh for p in self.pipelines]

    @property
    def n_nodes(self) -> list[int]:
        """Node counts per base coil."""
        return [m.points.shape[0] for m in self.meshes]

    @property
    def n_cells(self) -> list[int]:
        """Cell counts per base coil."""
        return [m.cells.shape[0] for m in self.meshes]
        
    # ============================================================================
    # Visualisation
    # ============================================================================

    def save_support_vtu(
        self,
        out_dir: str = ".",
        *,
        prefix: str = "coil",
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: dict | None = None,
    ) -> list[str]:
        """Export Winkler support weights and full mesh as VTU files.

        Writes one ``{prefix}{i:02d}_support.vtu`` per coil with point fields
        ``w_clamp``, ``w_attach``, ``k_clamp_Npm3``, and ``k_attach_Npm3``.
        :class:`~coil_fem.coupling.SupportBeams` additionally writes
        ``{prefix}_beams.vtu`` when ``base_support_dofs`` is provided — one
        line per beam on the free span ``[ξ_start, ξ_end]`` with cell field
        ``beam_length`` equal to ``L_eff``.
        :class:`~coil_fem.coupling.SupportBeamsCSR` additionally writes
        ``{prefix}_csr.vtu`` with the same weight point fields on the ring.

        Parameters
        ----------
        out_dir : str
            Output directory.  Created if it does not exist.
        prefix : str
            File-name prefix (default ``"coil"``).
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``self.base_curves_jax``.
        base_support_dofs : dict or None
            Per-coil support parameters for the support functions.

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]

        curves_jax = self.curves_from_dofs(base_curves_dofs)
        k_clamp = float(self.support.k_clamp)
        k_attach = float(self.support.k_attachment)

        os.makedirs(out_dir, exist_ok=True)
        written: list[str] = []

        for i, coil_mesh in enumerate(self.meshes):
            pts_i   = coil_mesh.mesh_points_from_dofs(base_curves_dofs[i])
            pts_np  = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(
                self.pipelines[i].surface_node_indices, dtype=onp.int32
            )
            w_g, w_a = self._support_weights(
                i, pts_i, curves_jax, base_support_dofs, at='nodes',
            )
            w_g_full = onp.zeros(n_nodes, dtype=onp.float64)
            w_a_full = onp.zeros(n_nodes, dtype=onp.float64)
            w_g_full[surf_idx] = onp.asarray(w_g, dtype=onp.float64)
            w_a_full[surf_idx] = onp.asarray(w_a, dtype=onp.float64)

            mesh_path = os.path.join(out_dir, f"{prefix}{i:02d}_support.vtu")
            meshio.Mesh(
                points=pts_np,
                cells=[(coil_mesh.meshio_cell_type, onp.asarray(coil_mesh.cells, dtype=onp.int32))],
                point_data={
                    "w_clamp": w_g_full,
                    "w_attach": w_a_full,
                    "k_clamp_Npm3": w_g_full * k_clamp,
                    "k_attach_Npm3": w_a_full * k_attach,
                },
            ).write(mesh_path)
            written.append(mesh_path)

        if base_support_dofs is not None and hasattr(self.support, 'beam_segments'):
            # Free-segment endpoints (surface-to-surface / surface-to-foundation).
            geom = self.support.beam_geometry(curves_jax, base_support_dofs)
            x_s = onp.asarray(geom['x_start'], dtype=onp.float64)
            x_e = onp.asarray(geom['x_end'], dtype=onp.float64)
            xi_s = onp.asarray(geom['xi_start'], dtype=onp.float64)[:, None]
            xi_e = onp.asarray(geom['xi_end'], dtype=onp.float64)[:, None]
            x_att_s = x_s + xi_s * (x_e - x_s)
            x_att_e = x_s + xi_e * (x_e - x_s)
            n_beams = x_s.shape[0]

            pts_beams = onp.concatenate([x_att_s, x_att_e], axis=0)  # (2N, 3)
            conn = onp.column_stack([
                onp.arange(n_beams),
                onp.arange(n_beams, 2 * n_beams),
            ]).astype(onp.int32)

            coil_idx_arr, beam_type = self.support.beam_labels()
            L_eff = onp.asarray(geom['L_eff'], dtype=onp.float64)

            beam_path = os.path.join(out_dir, f"{prefix}_beams.vtu")
            meshio.Mesh(
                points=pts_beams,
                cells=[("line", conn)],
                cell_data={
                    "beam_type":   [beam_type],
                    "beam_length": [L_eff],
                    "coil_index":  [coil_idx_arr],
                },
            ).write(beam_path)
            written.append(beam_path)
        else:
            geom = None

        if base_support_dofs is not None and hasattr(self.support, 'csr_mesh'):
            csr_mesh = self.support.csr_mesh
            csr_pts = onp.asarray(
                csr_mesh.mesh_points_from_dofs(
                    base_support_dofs['csr_curve_dofs']
                ),
                dtype=onp.float64,
            )
            pt_data = {}
            if hasattr(self.support, 'csr_attachment_weights'):
                if geom is None:
                    geom = self.support.beam_geometry(
                        curves_jax, base_support_dofs,
                    )
                surf_idx = onp.asarray(
                    self.support._csr_pipeline.surface_node_indices,
                    dtype=onp.int32,
                )
                w_g, w_a = self.support.csr_attachment_weights(
                    geom, base_support_dofs,
                )
                n_nodes = csr_pts.shape[0]
                w_g_full = onp.zeros(n_nodes, dtype=onp.float64)
                w_a_full = onp.zeros(n_nodes, dtype=onp.float64)
                w_g_full[surf_idx] = onp.asarray(w_g, dtype=onp.float64)
                w_a_full[surf_idx] = onp.asarray(w_a, dtype=onp.float64)
                pt_data = {
                    "w_clamp": w_g_full,
                    "w_attach": w_a_full,
                    "k_clamp_Npm3": w_g_full * k_clamp,
                    "k_attach_Npm3": w_a_full * k_attach,
                }
            csr_path = os.path.join(out_dir, f"{prefix}_csr.vtu")
            meshio.Mesh(
                points=csr_pts,
                cells=[(
                    csr_mesh.meshio_cell_type,
                    onp.asarray(csr_mesh.cells, dtype=onp.int32),
                )],
                point_data=pt_data,
            ).write(csr_path)
            written.append(csr_path)

        return written

    def plot_support(
        self,
        *,
        base_curves_dofs: list[jax.Array] | None = None,
        base_support_dofs: dict | None = None,
        ax=None,
        s: float = 0.1,
        cmap: str = "viridis",
        color="C0",
        simple_mode: bool = False,
        beam_color="k",
        beam_lw: float = 1.5,
        **kwargs,
    ):
        """Scatter-plot the mesh nodes of every base coil coloured by Winkler weight.

        For :class:`~coil_fem.coupling.SupportBeams`, also draws beam segments
        as line collections when ``base_support_dofs`` is provided.

        Parameters
        ----------
        base_curves_dofs : list[jax.Array] or None
            DOF vectors used to evaluate current surface positions.  ``None``
            uses the initial DOFs from ``self.base_curves_jax``.
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
        beam_color : color-like
            Colour of the beam line segments (default ``"k"``).  Ignored when
            the support has no beams.
        beam_lw : float
            Line width of the beam segments (default ``1.5``).
        **kwargs
            Extra keyword arguments forwarded to :meth:`ax.scatter`
            (e.g. ``marker``, ``facecolors``, ``edgecolors``).

        Returns
        -------
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
            The 3-D axes used for the plot.  The parent figure is available as
            ``ax.get_figure()``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb

        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]

        curves_jax = self.curves_from_dofs(base_curves_dofs)

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = ax.get_figure()

        sc = None
        for i, coil_mesh in enumerate(self.meshes):
            pts_i   = coil_mesh.mesh_points_from_dofs(base_curves_dofs[i])
            pts_np  = onp.asarray(pts_i, dtype=onp.float64)
            n_nodes = pts_np.shape[0]

            surf_idx = onp.asarray(
                self.pipelines[i].surface_node_indices, dtype=onp.int32
            )
            w_g, w_a = self._support_weights(
                i, pts_i, curves_jax, base_support_dofs, at='nodes',
            )
            weight_full = onp.zeros(n_nodes, dtype=onp.float64)
            weight_full[surf_idx] = onp.asarray(w_g + w_a, dtype=onp.float64)

            if simple_mode:
                rgba = onp.empty((n_nodes, 4), dtype=onp.float64)
                rgba[:, :3] = to_rgb(color)
                rgba[:, 3] = onp.clip(weight_full, 0.0, 1.0)
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

        if base_support_dofs is not None and hasattr(self.support, 'beam_segments'):
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            segs = self.support.beam_segments(curves_jax, base_support_dofs)  # (N, 2, 3)
            ax.add_collection3d(
                Line3DCollection(segs, colors=beam_color, linewidths=beam_lw)
            )

        return ax

    def plot(
        self,
        *,
        base_curves_dofs: list[jax.Array] | None = None,
        base_currents_dofs: jax.Array | None = None,
        base_support_dofs: dict | None = None,
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
        base_support_dofs : dict or None
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
        base_support_dofs: dict | None = None,
        n_sub: int = 20,
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
          - point fields ``w_clamp``, ``w_attach``, ``k_clamp_Npm3``, and
            ``k_attach_Npm3`` — grounded-clamp and beam-attachment weights and
            stiffnesses [N/m³].

        * ``{out_dir}/{prefix}_beams.vtu`` — polyline mesh of every base-coil
          beam on its free span ``[ξ_start, ξ_end]`` (surface-to-surface for
          CC, surface-to-foundation for CF), ``n_sub`` line segments
          (``n_sub + 1`` points) per beam, only when ``base_support_dofs`` is
          given and :attr:`support` is a
          :class:`~coil_fem.coupling.SupportBeams` (i.e. has a
          ``beam_displacement`` method):

          - point field ``displacement_m`` — closed-form beam-centreline
            displacement at each sub-point (see
            :meth:`~coil_fem.coupling.SupportBeams.beam_displacement`).
          - cell field ``beam_type`` (CC=0, CF=1), ``coil_index``, and
            ``beam_length`` (effective free length ``L_eff``).

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
        base_support_dofs : dict or None
            Per-coil support parameters for the support functions.
        n_sub : int
            Number of sub-segments per beam in ``{prefix}_beams.vtu``
            (default 20).  Ignored when the support has no beam network.

        Returns
        -------
        list[str]
            Paths of all files written, in order.
        """
        if base_curves_dofs is None:
            base_curves_dofs = [c.dofs for c in self.base_curves_jax]

        curves_jax = self.curves_from_dofs(base_curves_dofs)
        k_clamp = float(self.support.k_clamp)
        k_attach = float(self.support.k_attachment)

        result = self.run(
            base_curves_dofs=base_curves_dofs,
            base_currents_dofs=base_currents_dofs,
            base_support_dofs=base_support_dofs,
        )

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
            pts_i = jnp.asarray(pts_np)
            surf_idx = onp.asarray(
                self.pipelines[i].surface_node_indices, dtype=onp.int32
            )
            w_g, w_a = self._support_weights(
                i, pts_i, curves_jax, base_support_dofs, at='nodes',
            )
            w_g_full = onp.zeros(n_nodes, dtype=onp.float64)
            w_a_full = onp.zeros(n_nodes, dtype=onp.float64)
            w_g_full[surf_idx] = onp.asarray(w_g, dtype=onp.float64)
            w_a_full[surf_idx] = onp.asarray(w_a, dtype=onp.float64)
            pt_data = {
                "displacement_m": disp,
                "w_clamp": w_g_full,
                "w_attach": w_a_full,
                "k_clamp_Npm3": w_g_full * k_clamp,
                "k_attach_Npm3": w_a_full * k_attach,
            }

            mesh_path = os.path.join(out_dir, f"{prefix}{i:02d}_run.vtu")
            meshio.Mesh(
                points=pts_np,
                cells=[(coil_mesh.meshio_cell_type, onp.asarray(coil_mesh.cells, dtype=onp.int32))],
                point_data=pt_data,
                cell_data={
                    "von_mises_MPa": [vm_mpa],
                    "f_vol_Npm3":    [f_vol_cell],
                    "B_self_T":      [B_self_cell],
                    "B_ext_T":       [B_ext_cell],
                },
            ).write(mesh_path)
            written.append(mesh_path)

        # ── Beam free-span displacement (SupportBeams only) ───────────────────
        geom = None
        if (base_support_dofs is not None and result['u_s'] is not None
                and hasattr(self.support, 'beam_displacement')
                and self.support.n_beams_total > 0):
            geom = self.support.beam_geometry(curves_jax, base_support_dofs)
            xi_start = geom['xi_start']
            xi_end = geom['xi_end']
            # Per-beam uniform samples on the free chord [ξ_start, ξ_end].
            t = jnp.linspace(0.0, 1.0, n_sub + 1)
            xi = xi_start[:, None] + t[None, :] * (xi_end[:, None] - xi_start[:, None])
            disp = onp.asarray(
                self.support.beam_displacement(geom, result['u_s'], xi),
                dtype=onp.float64,
            )  # (N_beams, n_sub+1, 3)

            x_s = onp.asarray(geom['x_start'], dtype=onp.float64)  # (N_beams, 3)
            x_e = onp.asarray(geom['x_end'],   dtype=onp.float64)
            xi_np = onp.asarray(xi, dtype=onp.float64)
            pts = x_s[:, None, :] + xi_np[..., None] * (x_e - x_s)[:, None, :]

            n_beams, n_pts_per_beam, _ = pts.shape
            pts_flat  = pts.reshape(-1, 3)
            disp_flat = disp.reshape(-1, 3)

            base_idx = (onp.arange(n_beams) * n_pts_per_beam)[:, None]
            local    = onp.arange(n_sub)[None, :]
            conn = onp.stack([base_idx + local, base_idx + local + 1], axis=-1)
            conn = conn.reshape(-1, 2).astype(onp.int32)

            coil_idx_arr, beam_type = self.support.beam_labels()
            L_eff = onp.asarray(geom['L_eff'], dtype=onp.float64)
            beam_path = os.path.join(out_dir, f"{prefix}_beams.vtu")
            meshio.Mesh(
                points=pts_flat,
                cells=[("line", conn)],
                point_data={"displacement_m": disp_flat},
                cell_data={
                    "beam_type":   [onp.repeat(beam_type, n_sub)],
                    "coil_index":  [onp.repeat(coil_idx_arr, n_sub)],
                    "beam_length": [onp.repeat(L_eff, n_sub)],
                },
            ).write(beam_path)
            written.append(beam_path)

        if (base_support_dofs is not None and result['u_s'] is not None
                and hasattr(self.support, 'csr_displacement')):
            csr_mesh = self.support.csr_mesh
            csr_pts = onp.asarray(
                csr_mesh.mesh_points_from_dofs(
                    base_support_dofs['csr_curve_dofs']
                ),
                dtype=onp.float64,
            )
            csr_disp = onp.asarray(
                self.support.csr_displacement(result['u_s']),
                dtype=onp.float64,
            )
            pt_data = {"displacement_m": csr_disp}
            if hasattr(self.support, 'csr_attachment_weights'):
                if geom is None:
                    geom = self.support.beam_geometry(
                        curves_jax, base_support_dofs,
                    )
                surf_idx = onp.asarray(
                    self.support._csr_pipeline.surface_node_indices,
                    dtype=onp.int32,
                )
                w_g, w_a = self.support.csr_attachment_weights(
                    geom, base_support_dofs,
                )
                n_nodes = csr_pts.shape[0]
                w_g_full = onp.zeros(n_nodes, dtype=onp.float64)
                w_a_full = onp.zeros(n_nodes, dtype=onp.float64)
                w_g_full[surf_idx] = onp.asarray(w_g, dtype=onp.float64)
                w_a_full[surf_idx] = onp.asarray(w_a, dtype=onp.float64)
                pt_data.update({
                    "w_clamp": w_g_full,
                    "w_attach": w_a_full,
                    "k_clamp_Npm3": w_g_full * k_clamp,
                    "k_attach_Npm3": w_a_full * k_attach,
                })
            csr_path = os.path.join(out_dir, f"{prefix}_csr.vtu")
            meshio.Mesh(
                points=csr_pts,
                cells=[(
                    csr_mesh.meshio_cell_type,
                    onp.asarray(csr_mesh.cells, dtype=onp.int32),
                )],
                point_data=pt_data,
            ).write(csr_path)
            written.append(csr_path)

        return written
