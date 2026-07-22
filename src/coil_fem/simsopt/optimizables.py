"""Simsopt-optimizable coil support models.

A *support* describes where a coil is structurally clamped.  Each concrete
subclass of :class:`CoilSupport` holds a pure-functional
:class:`~coil_fem.coupling.Support` (or
:class:`~coil_fem.coupling.SupportBeams`) instance accessible via
:attr:`CoilSupport.support`.  That functional object is passed directly to
:class:`~coil_fem.CoilFEM`; the simsopt layer is responsible only for
maintaining the DOF state.

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
from jax.flatten_util import ravel_pytree
from ..presets import cross_section_fns
from ..utils import fetch_attr
from ..coupling import Support, SupportBeams, make_clamp_fn, make_topbottom_fn

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
    pytree).  The functional support model is stored in :attr:`support` and is
    passed directly to :class:`~coil_fem.CoilFEM`; subclasses do **not** need
    to inherit :class:`~coil_fem.coupling.Support` themselves.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (each exposing ``.curve`` and ``.current``).
        These are the *base* coils — before symmetry expansion.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry during the symmetry expansion.
    support : Support
        Pure-functional support instance (e.g. a bare
        :class:`~coil_fem.coupling.Support` with ``fixed_clamp_fns``, or a
        :class:`~coil_fem.coupling.SupportBeams`).  Passed verbatim to
        :class:`~coil_fem.CoilFEM` via :attr:`support`.
    support_dofs_jax : dict
        Optimisable support parameters.  Flattened into the simsopt DOF vector.
    constants : dict or None
        Fixed (non-optimised) scalars for introspection (e.g. ``r_clamp``).
        Not forwarded to the functional support; that object owns its own
        constants.
    names : list[str] or None
        Optional DOF names, length equal to the flattened ``support_dofs_jax``
        size.
    fixed : array-like or None
        Boolean fixed-mask aligned with the flattened DOF vector.
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
        support: Support,
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
        self._support = support

        flat, self._unravel = ravel_pytree(support_dofs_jax)
        self.constants = dict(constants or {})
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
    # Functional support accessor
    # ============================================================================

    @property
    def support(self) -> Support:
        """The pure-functional :class:`~coil_fem.coupling.Support` instance."""
        return self._support

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


def _broadcast_phis(phis, n_coils, n_clamp):
    if phis is None:
        row = jnp.linspace(0.0, 1.0, n_clamp, endpoint=False)
        phis_arr = jnp.broadcast_to(row, (n_coils, n_clamp))
    else:
        phis_arr = jnp.asarray(phis, dtype=float)
        if phis_arr.ndim == 1:
            phis_arr = jnp.broadcast_to(phis_arr, (n_coils, n_clamp))
    return phis_arr


class CoilSupportFixed(CoilSupport):
    """Support modelled as ``n_clamp`` clamps at optimisable arclengths.

    Each clamp is a sphere of radius ``r_clamp`` centred on the coil at
    curve parameter ``phi``; the per-node weight is the (smooth) union of all
    clamp indicator spheres.  Only the clamp locations ``phis`` are DOFs;
    ``r_clamp``, ``sigmoid_eps`` and ``n_clamp`` are fixed.

    Holds a merged ``phis`` array of shape ``(n_coils, n_clamp)`` covering
    all base coils; a single flat DOF vector of length
    ``n_coils * n_clamp`` is registered with simsopt.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (before symmetry expansion).
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry.
    fixed_clamp_options : dict
        Contains the following entries:
        r_clamp : float
            Sphere radius [m] of each clamp (region of non-zero spring weight).
        n_clamp : int
            Number of clamps per coil.
        sigmoid_eps : float
            Edge sharpness of the clamp (default 0.1).
    phis : array-like or None
        Initial clamp locations, shape ``(n_coils, n_clamp)`` or
        ``(n_clamp,)`` (broadcast to all coils).  ``None`` (default) spreads
        ``n_clamp`` clamps uniformly via ``linspace(0, 1, n_clamp,
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
        fixed_clamp_options: dict,
        phis=None,
        names=None,
        dofs=None,
    ):
        n_coils = len(base_coils)

        try:
            r_clamp = fixed_clamp_options['r_clamp']
            n_clamp = fixed_clamp_options['n_clamp']
        except KeyError:
            raise KeyError(
                "fixed_clamp_options must contain 'r_clamp' and 'n_clamp'."
            )
        sigmoid_eps = fixed_clamp_options.get('sigmoid_eps', 0.1)

        phis_arr = _broadcast_phis(phis, n_coils, n_clamp)

        # Build one closure per coil so Support.compute_weights can dispatch
        # correctly even when the phis come from the simsopt DOF store.
        fns = [
            make_clamp_fn(i, r_clamp, sigmoid_eps, phis_arr[i])
            for i in range(n_coils)
        ]

        super().__init__(
            base_coils, nfp, stellsym,
            support=Support(fixed_clamp_fns=fns),
            support_dofs_jax={'phis': phis_arr},
            constants={
                'r_clamp': float(r_clamp),
                'sigmoid_eps': float(sigmoid_eps),
            },
            names=names,
            dofs=dofs,
        )


class CoilSupportTopBottom(CoilSupport):
    """Static soft-sphere support at the top and bottom of the coil centreline.

    Has no optimisable DOFs (``support_dofs_jax={}``); ``r_clamp`` and
    ``sigmoid_eps`` are fixed constants.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (before symmetry expansion).
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether to apply stellarator symmetry.
    fixed_clamp_options : dict
        Contains the following entries:
        r_clamp : float
            Sphere radius [m] of each clamp (region of non-zero spring weight).
        sigmoid_eps : float
            Edge sharpness of the clamp (default 0.1).
    dofs : DOFs or None
        Simsopt ``DOFs`` object for restoring serialised state.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        fixed_clamp_options: dict,
        dofs=None,
    ):
        try:
            r_clamp = fixed_clamp_options['r_clamp']
        except KeyError:
            raise KeyError("fixed_clamp_options must contain 'r_clamp'.")
        sigmoid_eps = fixed_clamp_options.get('sigmoid_eps', 0.1)

        super().__init__(
            base_coils, nfp, stellsym,
            support=Support(fixed_clamp_fns=make_topbottom_fn(float(r_clamp), float(sigmoid_eps))),
            support_dofs_jax={},
            constants={
                'r_clamp': float(r_clamp),
                'sigmoid_eps': float(sigmoid_eps),
            },
            dofs=dofs,
        )


_REQUIRED_BEAM_OPTIONS = (
    'n_beam_cc',
    'n_beam_cf',
    'E',
    'nu',
    'k_lin',
    'k_tor',
)


class CoilSupportBeams(CoilSupport):
    """Simsopt-optimizable beam-network coil support.

    Wraps :class:`~coil_fem.coupling.SupportBeams` as a simsopt
    :class:`~simsopt._core.optimizable.Optimizable`.  The beam attachment
    angles (``phis_start_cc``, ``phis_end_cc``, ``phis_start_cf``), cross-section
    orientations (``thetas_orientation_cc``, ``thetas_orientation_cf``), and
    foundation anchor positions (``x_foundation``) are the optimisable DOFs.

    The underlying :class:`~coil_fem.coupling.SupportBeams` instance is
    accessible via :attr:`~CoilSupport.support`.

    Parameters
    ----------
    base_coils : list
        Simsopt ``Coil`` objects (before symmetry expansion); each must expose
        ``.curve`` (a simsopt ``CurveXYZFourier``) and ``.current``.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether stellarator symmetry is applied.
    beam_options : dict
        Contains the following entries:
        n_beam_cc : int
            Number of coil-coil beams per base coil.
        n_beam_cf : int
            Number of coil-foundation beams per base coil.
        E : float
            Young's modulus [Pa].
        nu : float
            Poisson ratio.
        cross_section_type : str
            Cross section shape. Must choose from the functions available in
            coil_fem.presets.cross_section_fns.
        attachment_type : callable
            ``clamp_fn(surface_pts_beam_frame, dofs, sign_x) -> weights``;
            selects coil surface nodes for beam endpoint coupling.
        k_lin : float
            Translational spring stiffness [N/m²].
        k_tor : float
            Torsional spring stiffness [N·m/m²].
    phis_start_cc : array-like or None
        Initial start-attachment angles for CC beams, shape
        ``(n_base, n_beam_cc)``.
    phis_end_cc : array-like or None
        Initial end-attachment angles for CC beams, shape
        ``(n_base, n_beam_cc)``.
    phis_start_cf : array-like or None
        Initial start-attachment angles for CF beams, shape
        ``(n_base, n_beam_cf)``.
    x_foundation : array-like or None
        Initial foundation anchor positions, shape ``(n_base, n_beam_cf, 3)``.
    thetas_orientation_cc : array-like or None
        Initial cross-section roll angles for CC beams, shape
        ``(n_base, n_beam_cc)``.
    thetas_orientation_cf : array-like or None
        Initial cross-section roll angles for CF beams, shape
        ``(n_base, n_beam_cf)``.
    fixed_clamp_options : dict
        Optional additional fixed-sphere Winkler clamps on the coil surface.
        Set ``{'enabled': True, 'r_clamp': ..., 'n_clamp': ...}`` to enable;
        ``{'enabled': False}`` (default) disables.
    phis : array-like or None
        Initial clamp locations for the optional fixed-sphere clamps.
    fixed_dof_names : iterable of str or None
        Names of ``support_dofs`` keys whose values should be **fixed** (not
        optimised).
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
        # Beam info
        beam_options={},
        phis_start_cc=None,
        phis_end_cc=None,
        phis_start_cf=None,
        x_foundation=None,
        thetas_orientation_cc=None,
        thetas_orientation_cf=None,
        # Clamp info
        fixed_clamp_options={'enabled': False},
        phis=None,
        # Simsopt info
        fixed_dof_names=None,
        names=None,
        dofs=None,
        **kwargs,
    ):
        from ..geo import CurveXYZFourierJAX

        # ── Convert simsopt Coil objects to JAX curve objects ─────────────────
        base_curves_jax = [
            CurveXYZFourierJAX.from_simsopt(c.curve) for c in base_coils
        ]
        n_base = len(base_coils)

        # ── Load beam options ─────────────────────────────────────────────────
        missing_beam_options = [k for k in _REQUIRED_BEAM_OPTIONS if k not in beam_options]
        if missing_beam_options:
            raise ValueError(
                f"beam_options must contain {missing_beam_options}."
            )
        n_beam_cc = beam_options['n_beam_cc']
        n_beam_cf = beam_options['n_beam_cf']
        E = beam_options['E']
        nu = beam_options['nu']
        k_lin = beam_options['k_lin']
        k_tor = beam_options['k_tor']
        cross_section_type = beam_options.get('cross_section_type', 'solid_circle')
        attachment_type = beam_options.get('attachment_type', 'direct')

        # ── Cross-section preset ──────────────────────────────────────────────
        cross_section_fn = fetch_attr(cross_section_type, cross_section_fns)
        cross_section_fn_keys = fetch_attr(cross_section_type + '_keys', cross_section_fns)

        # ── Build initial support_dofs_jax with defaults ──────────────────────
        _uniform_cc = jnp.broadcast_to(
            jnp.linspace(0., 1., n_beam_cc, endpoint=False),
            (n_base, n_beam_cc),
        )
        _uniform_cf = (
            jnp.broadcast_to(
                jnp.linspace(0., 1., n_beam_cf, endpoint=False),
                (n_base, n_beam_cf),
            ) if n_beam_cf > 0 else jnp.zeros((n_base, n_beam_cf))
        )

        support_dofs_jax = {
            'phis_end_cc': (
                _uniform_cc if phis_end_cc is None
                else jnp.asarray(phis_end_cc, dtype=float)
            ),
            'phis_start_cc': (
                _uniform_cc if phis_start_cc is None
                else jnp.asarray(phis_start_cc, dtype=float)
            ),
            'phis_start_cf': (
                _uniform_cf if phis_start_cf is None
                else jnp.asarray(phis_start_cf, dtype=float)
            ),
            'thetas_orientation_cc': (
                jnp.zeros((n_base, n_beam_cc)) if thetas_orientation_cc is None
                else jnp.asarray(thetas_orientation_cc, dtype=float)
            ),
            'thetas_orientation_cf': (
                jnp.zeros((n_base, n_beam_cf)) if thetas_orientation_cf is None
                else jnp.asarray(thetas_orientation_cf, dtype=float)
            ),
            'x_foundation': (
                jnp.zeros((n_base, n_beam_cf, 3)) if x_foundation is None
                else jnp.asarray(x_foundation, dtype=float)
            ),
        }

        # Cross-section DOF keys (e.g. radius for solid_circle).
        if kwargs is None:
            raise AttributeError(
                "The cross section shape requires initial values of "
                f"'{cross_section_fn_keys}' as keyword arguments for "
                "CoilSupportBeams. No keyword arguments are detected."
            )
        for k in cross_section_fn_keys:
            if k in kwargs:
                support_dofs_jax[k] = kwargs[k]
            else:
                raise AttributeError(
                    "The cross section shape requires an initial value of "
                    f"'{k}' as keyword argument for CoilSupportBeams."
                )

        # ── SupportBeams constants (owned by the functional object) ───────────
        beam_constants = {
            'nfp':        nfp,
            'stellsym':   stellsym,
            'n_beam_cc':  n_beam_cc,
            'n_beam_cf':  n_beam_cf,
            'E':          E,
            'nu':         nu,
            'k_lin':      k_lin,
            'k_tor':      k_tor,
        }

        # ── Optional fixed-sphere Winkler clamps ──────────────────────────────
        if fixed_clamp_options.get('enabled', False):
            try:
                r_clamp = fixed_clamp_options['r_clamp']
                n_clamp = fixed_clamp_options['n_clamp']
            except KeyError:
                raise KeyError(
                    "fixed_clamp_options must contain 'r_clamp' and 'n_clamp'."
                )
            sigmoid_eps = fixed_clamp_options.get('sigmoid_eps', 0.1)

            phis_arr = _broadcast_phis(phis, n_base, n_clamp)
            support_dofs_jax['phis'] = phis_arr

            fixed_clamp_fns = [
                make_clamp_fn(i, r_clamp, sigmoid_eps, phis_arr[i])
                for i in range(n_base)
            ]
        else:
            if n_beam_cf == 0:
                raise AttributeError(
                    "The coils must be supported by either fixed clamps or "
                    "coil-foundation beams. Currently neither is enabled."
                )
            fixed_clamp_fns = None

        # ── Attachment function preset ─────────────────────────────────────────
        if attachment_type == 'direct':
            attachment_fn = fetch_attr(cross_section_type + '_attachment', cross_section_fns)
        elif attachment_type == 'wrap':
            attachment_fn = fetch_attr('wrap_attachment', cross_section_fns)

        # ── Build the functional SupportBeams ─────────────────────────────────
        beams = SupportBeams(
            constants=beam_constants,
            base_curves_jax=base_curves_jax,
            cross_section_fn=cross_section_fn,
            attachment_fn=attachment_fn,
            fixed_clamp_fns=fixed_clamp_fns,
        )

        # ── Compute boolean fixed_mask from fixed_dof_names ───────────────────
        if fixed_dof_names is None:
            fixed_dof_names = list(cross_section_fn_keys) + \
                ['thetas_orientation_cc', 'thetas_orientation_cf']

        fixed_dof_names = list(fixed_dof_names)
        if n_beam_cc == 0:
            fixed_dof_names += ['phis_start_cc', 'phis_end_cc', 'thetas_orientation_cc']
        if n_beam_cf == 0:
            fixed_dof_names += ['phis_start_cf', 'x_foundation', 'thetas_orientation_cf']

        valid_keys = support_dofs_jax.keys()
        invalid_keys = [k for k in fixed_dof_names if k not in valid_keys]
        if invalid_keys:
            raise ValueError(
                f"fixed_dof_names: {invalid_keys} not valid "
                f"support_dofs key(s). Valid keys: {valid_keys}"
            )
        probe = {
            k: np.full_like(v, k in fixed_dof_names, dtype=bool)
            for k, v in support_dofs_jax.items()
        }
        fixed_mask, _ = ravel_pytree(probe)

        # ── Summary ───────────────────────────────────────────────────────────
        print('Beam network initialized.')
        print('Optimizing:')
        for k in valid_keys:
            if k not in fixed_dof_names:
                print('   ', k, '- shape:', np.array(support_dofs_jax[k]).shape)
        print('Fixing:')
        for k in fixed_dof_names:
            print('   ', k)
        print('Total # dofs:', len(fixed_mask) - np.sum(fixed_mask))

        # ── Initialize CoilSupport (calls Optimizable.__init__) ───────────────
        super().__init__(
            base_coils,
            nfp,
            stellsym,
            support=beams,
            support_dofs_jax=support_dofs_jax,
            constants=None,
            names=names,
            fixed=np.array(fixed_mask),
            dofs=dofs,
        )
