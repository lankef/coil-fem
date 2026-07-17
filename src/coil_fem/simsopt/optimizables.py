"""Simsopt-optimizable coil support models.

A *support* describes where a coil is structurally clamped.  Each concrete
subclass of :class:`CoilSupport` implements
:meth:`~coil_fem.coupling.Support.compute_weights`, which returns per-surface-node
Winkler spring weights in ``[0, 1]`` for a given base coil::

    weights = coil_support.compute_weights(
        coil_idx, surface_pts, curve_jax, support_dofs
    )

:class:`CoilSupport` holds the base coils (curves + currents), ``nfp``, and
``stellsym`` so that :class:`~coil_fem.simsopt.CoilFEMObjective` needs only a
single ``coil_support`` argument to construct the full FEM pipeline.

The optimisable parameters live **only** in the simsopt DOF store: the dofs
dict is flattened to the Optimizable's ``x`` at construction, and
:attr:`CoilSupport.support_dofs` reconstructs the dict on demand from
``local_full_x`` (never cached).  Fixed scalars (e.g. clamp radius) are kept
in :attr:`CoilSupport.constants` and are not optimised.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.nn import sigmoid

from jax.flatten_util import ravel_pytree

from ..coupling import Support, SupportFixed, SupportBeams

try:
    from simsopt._core.optimizable import Optimizable
    _HAS_SIMSOPT = True
except ImportError:  # pragma: no cover
    Optimizable = object  # type: ignore[misc, assignment]
    _HAS_SIMSOPT = False


class CoilSupport(Optimizable):
    """Simsopt container for a coilset's support DOFs, constants, and coil refs.

    Holds the base coils (curves + currents), the symmetry parameters ``nfp``
    and ``stellsym``, and any optimisable support parameters (the ``dofs``
    pytree).  Subclasses must also inherit a concrete
    :class:`~coil_fem.coupling.Support` implementation (e.g.
    :class:`~coil_fem.coupling.SupportFixed` or
    :class:`~coil_fem.coupling.SupportBeams`) and call that class's
    ``__init__`` explicitly before calling ``super().__init__``.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (each exposing ``.curve`` and ``.current``).
        These are the *base* coils — before symmetry expansion.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry during the symmetry expansion.
    support_dofs_jax : dict
        Optimisable support parameters.  Flattened into the simsopt DOF vector.
    constants : dict or None
        Fixed (non-optimised) scalars forwarded to :meth:`compute_weights` via
        :attr:`constants`.
    names : list[str] or None
        Optional DOF names, length equal to the flattened ``support_dofs_jax``
        size.
    dofs : DOFs or None
        Simsopt ``DOFs`` object used to restore serialised state (passed by
        :meth:`from_dict` / :meth:`from_file`).  Leave as ``None`` when
        constructing fresh instances.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        support_dofs_jax: dict,
        constants: dict | None = None,
        names=None,
        fixed=None,
        dofs=None,
    ):
        if not _HAS_SIMSOPT:
            raise ImportError("simsopt is required for CoilSupport.")

        self._base_coils = list(base_coils)
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)

        flat, self._unravel = ravel_pytree(support_dofs_jax)
        self.constants = dict(constants or {})
        # np.array (not asarray) so the DOF buffer is writable: converting a
        # JAX array via np.asarray yields a read-only view.
        if dofs is not None:
            Optimizable.__init__(self, dofs=dofs, depends_on=self._base_coils)
        else:
            Optimizable.__init__(
                self,
                x0=np.array(flat, dtype=float),
                names=names,
                fixed=fixed,
                depends_on=self._base_coils,
            )

    # ============================================================================
    # Coil accessors
    # ============================================================================

    @property
    def base_curves(self) -> list:
        """Simsopt curve objects for the base coils (before symmetry expansion)."""
        return [c.curve for c in self._base_coils]

    @property
    def base_currents(self) -> list:
        """Simsopt current objects for the base coils."""
        return [c.current for c in self._base_coils]

    @property
    def n_coils(self) -> int:
        """Number of base coils."""
        return len(self._base_coils)

    # ============================================================================
    # DOF accessors
    # ============================================================================

    @property
    def support_dofs(self) -> dict:
        """Current dofs as a differentiable JAX pytree, read from the DOFs."""
        return self._unravel(jnp.asarray(self.local_full_x))

    def flatten_grad(self, grad_dofs: dict) -> np.ndarray:
        """Flatten a JAX gradient pytree into a simsopt DOF-aligned array."""
        return np.asarray(ravel_pytree(grad_dofs)[0], dtype=float)


class CoilSupportFixed(CoilSupport, SupportFixed):
    """Support modelled as ``clamp_num`` clamps at optimisable arclengths.

    Each clamp is a sphere of radius ``clamp_radius`` centred on the coil at
    curve parameter ``phi``; the per-node weight is the (smooth) union of all
    clamp indicator spheres.  Only the clamp locations ``phis`` are DOFs;
    ``clamp_radius``, ``sigmoid_eps`` and ``clamp_num`` are fixed.

    Holds a merged ``phis`` array of shape ``(n_coils, clamp_num)`` covering
    all base coils; a single flat DOF vector of length
    ``n_coils * clamp_num`` is registered with simsopt.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (before symmetry expansion).
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry.
    clamp_radius : float
        Sphere radius [m] of each clamp (region of non-zero spring weight).
    clamp_num : int
        Number of clamps per coil (default 2).
    sigmoid_eps : float
        Edge sharpness of the clamp (default 0.1).
    phis : array-like or None
        Initial clamp locations, shape ``(n_coils, clamp_num)`` or
        ``(clamp_num,)`` (broadcast to all coils).  ``None`` (default) spreads
        ``clamp_num`` clamps uniformly via ``linspace(0, 1, clamp_num,
        endpoint=False)`` for every coil.
    names : list[str] or None
        Optional DOF names.
    dofs : DOFs or None
        Simsopt ``DOFs`` object for restoring serialised state.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        clamp_radius: float,
        clamp_num: int = 2,
        sigmoid_eps: float = 0.1,
        phis=None,
        names=None,
        dofs=None,
    ):
        SupportFixed.__init__(self, support_fns=None)

        n_coils = len(base_coils)
        clamp_radius = float(clamp_radius)
        clamp_num = int(clamp_num)

        if phis is None:
            row = jnp.linspace(0.0, 1.0, clamp_num, endpoint=False)
            phis_arr = jnp.broadcast_to(row, (n_coils, clamp_num))
        else:
            phis_arr = jnp.asarray(phis, dtype=float)
            if phis_arr.ndim == 1:
                phis_arr = jnp.broadcast_to(phis_arr, (n_coils, clamp_num))

        self.clamp_radius = clamp_radius
        self.clamp_num    = clamp_num
        self.sigmoid_eps  = float(sigmoid_eps)
        self.phis         = phis_arr
        self.names        = names

        super().__init__(
            base_coils, nfp, stellsym,
            support_dofs_jax={'phis': phis_arr},
            constants={
                'clamp_radius': clamp_radius,
                'sigmoid_eps': float(sigmoid_eps),
            },
            names=names,
            dofs=dofs,
        )

    @staticmethod
    def support_fn(surface_points, curve_jax, phis_i, *, clamp_radius, sigmoid_eps):
        """Per-surface-node Winkler weights for one coil: smooth union of clamp spheres.

        Parameters
        ----------
        surface_points : jax.Array, shape (n_surface_nodes, 3)
        curve_jax : CurveXYZFourierJAX
        phis_i : jax.Array, shape (clamp_num,)
            Clamp arc-length locations for this coil.
        clamp_radius : float
        sigmoid_eps : float

        Returns
        -------
        jax.Array, shape (n_surface_nodes,)
        """
        gamma_support = curve_jax.gamma_eval(phis_i)            # (clamp_num, 3)
        distances = jnp.sum(
            (surface_points[:, None, :] - gamma_support[None, :, :]) ** 2,
            axis=-1,
        )                                           # (n_nodes, clamp_num)
        sigmoid_width = sigmoid_eps * clamp_radius
        w = sigmoid((clamp_radius**2 - distances) / (sigmoid_width**2))
        return jnp.sum(w, axis=-1)                 # union of clamps

    def compute_weights(self, coil_idx, surface_pts, curve_jax, dofs):
        """Winkler weights for base coil ``coil_idx``.

        Parameters
        ----------
        coil_idx : int
        surface_pts : jax.Array, shape ``(n_surface_nodes, 3)``
        curve_jax : CurveXYZFourierJAX
        dofs : dict or None
            Merged support dofs dict with key ``'phis'``, shape
            ``(n_coils, clamp_num)``.  ``None`` falls back to the initial
            ``phis`` stored at construction.

        Returns
        -------
        jax.Array, shape ``(n_surface_nodes,)``
        """
        if dofs is None:
            phis_i = self.phis[coil_idx]
        else:
            phis_i = dofs['phis'][coil_idx]
        return self.support_fn(
            surface_pts, curve_jax, phis_i,
            clamp_radius=self.constants['clamp_radius'],
            sigmoid_eps=self.constants['sigmoid_eps'],
        )


class CoilSupportTopBottom(CoilSupport, SupportFixed):
    """Static soft-sphere support at the top and bottom of the coil centreline.

    Has no optimisable DOFs (``support_dofs_jax={}``); ``clamp_radius`` and
    ``sigmoid_eps`` are fixed constants.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (before symmetry expansion).
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry.
    clamp_radius : float
        Sphere radius [m] around the top and bottom curve points.
    sigmoid_eps : float
        Edge sharpness of each sphere (default 0.1).
    dofs : DOFs or None
        Simsopt ``DOFs`` object for restoring serialised state.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        clamp_radius: float,
        sigmoid_eps: float = 0.1,
        dofs=None,
    ):
        SupportFixed.__init__(self, support_fns=None)

        self.clamp_radius = float(clamp_radius)
        self.sigmoid_eps  = float(sigmoid_eps)

        super().__init__(
            base_coils, nfp, stellsym,
            support_dofs_jax={},
            constants={
                'clamp_radius': float(clamp_radius),
                'sigmoid_eps': float(sigmoid_eps),
            },
            dofs=dofs,
        )

    @staticmethod
    def support_fn(surface_points, curve_jax, *, clamp_radius, sigmoid_eps):
        """Per-surface-node weights: soft union of top and bottom spheres.

        Parameters
        ----------
        surface_points : jax.Array, shape (n_surface_nodes, 3)
        curve_jax : CurveXYZFourierJAX
        clamp_radius : float
        sigmoid_eps : float

        Returns
        -------
        jax.Array, shape (n_surface_nodes,)
        """
        gamma  = curve_jax.gamma()                         # (n_quad, 3)
        top    = gamma[jnp.argmax(gamma[:, 2])]            # (3,) highest point
        bottom = gamma[jnp.argmin(gamma[:, 2])]            # (3,) lowest point

        d_top    = jnp.sum((surface_points - top)**2,    axis=-1)
        d_bottom = jnp.sum((surface_points - bottom)**2, axis=-1)

        sigmoid_width = sigmoid_eps * clamp_radius
        w_top    = sigmoid((clamp_radius**2 - d_top)   / (sigmoid_width**2))
        w_bottom = sigmoid((clamp_radius**2 - d_bottom) / (sigmoid_width**2))
        return w_top + w_bottom

    def compute_weights(self, coil_idx, surface_pts, curve_jax, dofs):
        """Winkler weights for base coil ``coil_idx``.

        Parameters
        ----------
        coil_idx : int
            Ignored (same top/bottom logic applied to every coil).
        surface_pts : jax.Array, shape ``(n_surface_nodes, 3)``
        curve_jax : CurveXYZFourierJAX
        dofs : dict or None
            Ignored (no optimisable DOFs).

        Returns
        -------
        jax.Array, shape ``(n_surface_nodes,)``
        """
        return self.support_fn(
            surface_pts, curve_jax,
            clamp_radius=self.constants['clamp_radius'],
            sigmoid_eps=self.constants['sigmoid_eps'],
        )


class CoilSupportBeams(CoilSupport, SupportBeams):
    """Simsopt-optimizable beam-network coil support.

    Wraps :class:`~coil_fem.coupling.SupportBeams` as a simsopt
    :class:`~simsopt._core.optimizable.Optimizable`.  The beam attachment
    angles (``phi_start_cc``, ``phi_end_cc``, ``phi_start_cf``), cross-section
    orientations (``theta_orientation_cc``, ``theta_orientation_cf``), and
    foundation anchor positions (``x_foundation``) are the optimisable DOFs.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (before symmetry expansion); each must expose
        ``.curve`` (a simsopt ``CurveXYZFourier``) and ``.current``.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether stellarator symmetry is applied.
    n_beam_cc : int
        Number of coil-coil beams per base coil.
    n_beam_cf : int
        Number of coil-foundation beams per base coil.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio.
    cross_section_fn : callable
        ``cross_section_fn(support_dofs) -> (A, Iy, Iz, J)``; shapes
        ``(n_base, n_beam_cc + n_beam_cf)`` each.
    clamp_fn : callable
        ``clamp_fn(surface_pts, curve_jax, dofs, direction) -> weights``;
        selects coil surface nodes for beam endpoint coupling.
    k_lin : float
        Translational spring stiffness [N/m²].
    k_tor : float
        Torsional spring stiffness [N·m/m²].
    phi_start_cc : array-like or None
        Initial start-attachment angles for CC beams, shape
        ``(n_base, n_beam_cc)``.  ``None`` defaults to uniform
        ``linspace(0, 1, n_beam_cc, endpoint=False)`` broadcast to all coils.
    phi_end_cc : array-like or None
        Initial end-attachment angles for CC beams, shape
        ``(n_base, n_beam_cc)``.  Same default as ``phi_start_cc``.
    phi_start_cf : array-like or None
        Initial start-attachment angles for CF beams, shape
        ``(n_base, n_beam_cf)``.  Same uniform default.
    x_foundation : array-like or None
        Initial foundation anchor positions, shape ``(n_base, n_beam_cf, 3)``.
        ``None`` defaults to zeros.
    theta_orientation_cc : array-like or None
        Initial cross-section roll angles for CC beams, shape
        ``(n_base, n_beam_cc)``.  ``None`` defaults to zeros.
    theta_orientation_cf : array-like or None
        Initial cross-section roll angles for CF beams, shape
        ``(n_base, n_beam_cf)``.  ``None`` defaults to zeros.
    support_fns : callable or list[callable] or None
        Optional Winkler weight functions forwarded to
        :class:`~coil_fem.coupling.SupportFixed`.  When set,
        :meth:`compute_weights` behaves identically to
        :class:`~coil_fem.coupling.SupportFixed`.
    fixed_support_dofs_keys : iterable of str
        Names of ``support_dofs`` keys whose values should be **fixed** (not
        optimised).  Valid keys: ``'phi_end_cc'``, ``'phi_start_cc'``,
        ``'phi_start_cf'``, ``'theta_orientation_cc'``,
        ``'theta_orientation_cf'``, ``'x_foundation'``.  Each named key is
        fixed to its initial value; all remaining keys are free DOFs.
    names : list[str] or None
        Optional DOF names for the full (free + fixed) DOF vector.
    dofs : DOFs or None
        Simsopt ``DOFs`` object for restoring serialised state.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        n_beam_cc: int,
        n_beam_cf: int,
        E: float,
        nu: float,
        cross_section_fn,
        clamp_fn,
        k_lin: float,
        k_tor: float,
        phi_start_cc=None,
        phi_end_cc=None,
        phi_start_cf=None,
        x_foundation=None,
        theta_orientation_cc=None,
        theta_orientation_cf=None,
        support_fns=None,
        fixed_support_dofs_keys=(),
        names=None,
        dofs=None,
    ):
        from ..geo import CurveXYZFourierJAX

        # ── Convert simsopt Coil objects to JAX curve objects ─────────────────
        base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c.curve) for c in base_coils
        ]
        n_base = len(base_coils)

        # ── Initialise SupportBeams (also calls SupportFixed.__init__) ────────
        SupportBeams.__init__(
            self,
            base_curves_jax=base_curves_jax,
            nfp=nfp,
            stellsym=stellsym,
            n_beam_cc=n_beam_cc,
            n_beam_cf=n_beam_cf,
            E=E,
            nu=nu,
            cross_section_fn=cross_section_fn,
            clamp_fn=clamp_fn,
            k_lin=k_lin,
            k_tor=k_tor,
            support_fns=support_fns,
        )

        # ── Build initial support_dofs_jax with defaults ──────────────────────
        # Keys are listed alphabetically — this matches the ravel_pytree sort
        # order and makes the fixed-mask computation easy to verify.
        _uniform_cc = jnp.broadcast_to(
            jnp.linspace(0., 1., n_beam_cc, endpoint=False),
            (n_base, n_beam_cc),
        )
        _uniform_cf = jnp.broadcast_to(
            jnp.linspace(0., 1., n_beam_cf, endpoint=False),
            (n_base, n_beam_cf),
        ) if n_beam_cf > 0 else jnp.zeros((n_base, n_beam_cf))

        support_dofs_jax = {
            'phi_end_cc': (
                _uniform_cc if phi_end_cc is None
                else jnp.asarray(phi_end_cc, dtype=float)
            ),
            'phi_start_cc': (
                _uniform_cc if phi_start_cc is None
                else jnp.asarray(phi_start_cc, dtype=float)
            ),
            'phi_start_cf': (
                _uniform_cf if phi_start_cf is None
                else jnp.asarray(phi_start_cf, dtype=float)
            ),
            'theta_orientation_cc': (
                jnp.zeros((n_base, n_beam_cc)) if theta_orientation_cc is None
                else jnp.asarray(theta_orientation_cc, dtype=float)
            ),
            'theta_orientation_cf': (
                jnp.zeros((n_base, n_beam_cf)) if theta_orientation_cf is None
                else jnp.asarray(theta_orientation_cf, dtype=float)
            ),
            'x_foundation': (
                jnp.zeros((n_base, n_beam_cf, 3)) if x_foundation is None
                else jnp.asarray(x_foundation, dtype=float)
            ),
        }

        # ── Compute boolean fixed_mask from fixed_support_dofs_keys ──────────
        # The probe-dict technique: ravel a version of the dict where only
        # the target key is non-zero, then find those flat indices.
        flat, _ = ravel_pytree(support_dofs_jax)
        fixed_mask = np.zeros(len(flat), dtype=bool)
        valid_keys = set(support_dofs_jax)
        for key in fixed_support_dofs_keys:
            if key not in valid_keys:
                raise ValueError(
                    f"fixed_support_dofs_keys: '{key}' is not a valid "
                    f"support_dofs key. Valid keys: {sorted(valid_keys)}"
                )
            probe = {k: jnp.zeros_like(v) for k, v in support_dofs_jax.items()}
            probe[key] = jnp.ones_like(support_dofs_jax[key])
            probe_flat, _ = ravel_pytree(probe)
            fixed_mask[np.where(np.asarray(probe_flat) != 0)[0]] = True

        # ── Initialise CoilSupport (calls Optimizable.__init__) ───────────────
        super().__init__(
            base_coils,
            nfp,
            stellsym,
            support_dofs_jax=support_dofs_jax,
            constants={
                'n_beam_cc': n_beam_cc,
                'n_beam_cf': n_beam_cf,
                'k_lin':     k_lin,
                'k_tor':     k_tor,
            },
            names=names,
            fixed=fixed_mask,
            dofs=dofs,
        )
