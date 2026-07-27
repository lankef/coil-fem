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
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from ..presets import cross_section_fns
from ..utils import clamp_sigmoid
from ..coupling import Support, SupportBeams


# ============================================================================
# Clamp weight helpers (shape-specific, used by CoilSupportFixed/TopBottom)
# ============================================================================

def _clamp_spheres_weights(
    surface_points: jax.Array,
    curve_jax,
    phis_i: jax.Array,
    r_clamp: float,
    sigmoid_eps: float,
) -> jax.Array:
    """Smooth-union Winkler weight for a set of sphere clamps on one coil.

    Parameters
    ----------
    surface_points : jax.Array, shape (n_nodes, 3)
    curve_jax : CurveXYZFourierJAX
    phis_i : jax.Array, shape (n_clamp,)
        Arc-length locations of the clamps (values in [0, 1]).
    r_clamp : float
        Sphere radius [m].
    sigmoid_eps : float
        Sharpness of the clamp boundary.

    Returns
    -------
    jax.Array, shape (n_nodes,)
    """
    gamma_support = curve_jax.gamma_eval(phis_i)            # (n_clamp, 3)
    d_sq = jnp.sum(
        (surface_points[:, None, :] - gamma_support[None, :, :]) ** 2,
        axis=-1,
    )                                                        # (n_nodes, n_clamp)
    w = clamp_sigmoid(d_sq, r_clamp, sigmoid_eps)
    return jnp.sum(w, axis=-1)


def _topbottom_weights(
    surface_points: jax.Array,
    curve_jax,
    r_clamp: float,
    sigmoid_eps: float,
) -> jax.Array:
    """Winkler weight from soft spheres at the coil's topmost and bottommost points.

    Parameters
    ----------
    surface_points : jax.Array, shape (n_nodes, 3)
    curve_jax : CurveXYZFourierJAX
    r_clamp : float
    sigmoid_eps : float

    Returns
    -------
    jax.Array, shape (n_nodes,)
    """
    gamma  = curve_jax.gamma()
    top    = gamma[jnp.argmax(gamma[:, 2])]
    bottom = gamma[jnp.argmin(gamma[:, 2])]

    w_top = clamp_sigmoid(
        jnp.sum((surface_points - top) ** 2, axis=-1),
        r_clamp, sigmoid_eps,
    )
    w_bottom = clamp_sigmoid(
        jnp.sum((surface_points - bottom) ** 2, axis=-1),
        r_clamp, sigmoid_eps,
    )
    return w_top + w_bottom

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

        self._r_clamp = float(r_clamp)
        self._sig_eps  = float(sigmoid_eps)

        # ── Stored for GSONable serialization ────────────────────────────────────
        # GSONable.as_dict requires every __init__ parameter to be present as
        # self.<param> or self._<param>.
        self._fixed_clamp_options = {
            'r_clamp': r_clamp, 'n_clamp': n_clamp, 'sigmoid_eps': sigmoid_eps,
        }
        # Serialization seeds only — actual clamp positions are restored from the
        # Optimizable dofs object by simsopt on load.
        self._phis  = phis
        self._names = names
        # ─────────────────────────────────────────────────────────────────────────

        super().__init__(
            base_coils, nfp, stellsym,
            support=Support(fixed_clamp_fns=self._clamp_fn),
            support_dofs_jax={'phis': phis_arr},
            constants={
                'r_clamp': float(r_clamp),
                'sigmoid_eps': float(sigmoid_eps),
            },
            names=names,
            dofs=dofs,
        )

    def _clamp_fn(self, surface_pts, curve_jax, dofs_slice):
        """Winkler weight for one coil; dispatched by Support.compute_weights."""
        if not dofs_slice:
            return jnp.zeros(surface_pts.shape[0])
        return _clamp_spheres_weights(
            surface_pts, curve_jax, dofs_slice['phis'],
            self._r_clamp, self._sig_eps,
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

        self._r_clamp = float(r_clamp)
        self._sig_eps  = float(sigmoid_eps)

        # ── Stored for GSONable serialization ────────────────────────────────────
        # GSONable.as_dict requires every __init__ parameter to be present as
        # self.<param> or self._<param>.
        self._fixed_clamp_options = {'r_clamp': r_clamp, 'sigmoid_eps': sigmoid_eps}
        # ─────────────────────────────────────────────────────────────────────────

        super().__init__(
            base_coils, nfp, stellsym,
            support=Support(fixed_clamp_fns=self._clamp_fn),
            support_dofs_jax={},
            constants={
                'r_clamp': float(r_clamp),
                'sigmoid_eps': float(sigmoid_eps),
            },
            dofs=dofs,
        )

    def _clamp_fn(self, surface_pts, curve_jax, dofs_slice):
        """Winkler weight at the coil's topmost and bottommost points."""
        return _topbottom_weights(surface_pts, curve_jax, self._r_clamp, self._sig_eps)


_REQUIRED_BEAM_OPTIONS = (
    'n_beam_cc',
    'n_beam_cf',
    'E',
    'nu',
    'k_attachment',
)


# ============================================================================
# CoilSupportBeams construction helpers (used only during __init__)
# ============================================================================

def _uniform_list(counts, cc_stellsym=False, cc_end=False):
    """Build a per-group list of uniformly spaced initial phi values.

    Used during :class:`CoilSupportBeams` construction to initialise
    ``phis_start_cc``, ``phis_end_cc``, and ``phis_start_cf`` when the caller
    does not supply explicit values.

    When ``cc_stellsym=True``, the last two entries (the stellsym wrap group
    and its reflection) use a restricted range ``[0, 0.5)`` to avoid beam
    overlap after the stellarator reflection.
    """
    out = []
    for i in range(len(counts)):
        c = counts[i]
        if c == 0:
            out.append(jnp.array([]))
            continue
        if not cc_stellsym or i < len(counts) - 2:
            a, b = 0.0, 1.0
            half_interval = 0.5 / c
        else:
            a, b = 0.0, 0.5
            half_interval = 0.5 / c / 2
        phi_init = jnp.linspace(a, b, c, endpoint=False) + half_interval
        if cc_end and i >= len(counts) - 2:
            phi_init = 1 - phi_init
        out.append(phi_init)
    return out


def _zeros_list(counts, trailing=()):
    """Build a per-group list of zero arrays with optional trailing shape."""
    return [jnp.zeros((counts[i],) + trailing) for i in range(len(counts))]


def _check_ragged_shape(value, counts, name, trailing=()):
    """Validate and cast a user-supplied ragged DOF to a per-group list.

    Returns ``None`` when ``value`` is ``None`` (caller should substitute a
    default).  Raises ``ValueError`` when length or per-entry shapes disagree
    with ``counts``.
    """
    if value is None:
        return None
    seq = list(value)
    if len(seq) != len(counts):
        raise ValueError(
            f"{name} must be a length-{len(counts)} sequence (one entry "
            f"per CC group for cc keys — n_base + 1 when stellsym=True "
            f"— or per base coil for cf keys); got length {len(seq)}."
        )
    out = []
    for i, v in enumerate(seq):
        arr = jnp.asarray(v, dtype=float)
        expected = (counts[i],) + trailing
        if arr.shape != expected:
            raise ValueError(
                f"{name}[{i}] must have shape {expected}; got {arr.shape}."
            )
        out.append(arr)
    return out


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
        n_beam_cc : int or sequence of int
            Number of coil-coil beams per CC group.  A scalar is broadcast
            to every group; a sequence must have one entry per group:
            ``n_base + 1`` entries when ``stellsym=True`` (the extra last
            entry is the coil-0 ``phi = 0`` wrap group), else ``n_base``.
        n_beam_cf : int or sequence of int
            Number of coil-foundation beams per base coil.  Scalar (broadcast)
            or length-``n_base`` sequence.
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
        k_attachment : float
            Distributed attachment (Winkler) modulus [N/m³].  Governs both
            translational and rotational spring coupling.  Must equal
            ``problem_options['winkler_k']``.
        sigmoid_eps : float
            Sigmoid function widths of attachment points. Default is 0.1.
        and additional options needed by attachment_fn.
    phis_start_cc : sequence of array-like or None
        Initial start-attachment angles for CC beams, one entry per CC group
        (``n_base + 1`` entries when ``stellsym=True``, else ``n_base``)
        with entry ``g`` of shape ``(n_beam_cc[g],)``.
    phis_end_cc : sequence of array-like or None
        Initial end-attachment angles for CC beams, one entry per CC group
        with entry ``g`` of shape ``(n_beam_cc[g],)``.
    phis_start_cf : sequence of array-like or None
        Initial start-attachment angles for CF beams, a length-``n_base``
        sequence with entry ``i`` of shape ``(n_beam_cf[i],)``.
    x_foundation : sequence of array-like or None
        Initial foundation anchor positions, a length-``n_base`` sequence with
        entry ``i`` of shape ``(n_beam_cf[i], 3)``.
    thetas_orientation_cc : sequence of array-like or None
        Initial cross-section roll angles for CC beams, one entry per CC
        group with entry ``g`` of shape ``(n_beam_cc[g],)``.
    thetas_orientation_cf : sequence of array-like or None
        Initial cross-section roll angles for CF beams, a length-``n_base``
        sequence with entry ``i`` of shape ``(n_beam_cf[i],)``.
    fixed_clamp_options : dict
        Optional additional fixed-sphere Winkler clamps on the coil surface.
        Set ``{'enabled': True, 'r_clamp': ..., 'n_clamp': ...}`` to enable;
        ``{'enabled': False}`` (default) disables.
    phis : array-like or None
        Initial clamp locations for the optional fixed-sphere clamps.
    fixed_dof_names : iterable of str or None
        Names of ``support_dofs`` keys whose values should be **fixed** (not
        optimised). Some dofs will cause the problem to become ill-posed
        if not fixed/constrained. If wither of ``n_beam_cc`` or ``n_beam_cf`` 
        is 0, then all dofs associated with that type of beam will be fixed.
        By default, the attachment points of CF beams and the 
        parameters of each individual beam is fixed.
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
        beam_options=None,
        phis_start_cc=None,
        phis_end_cc=None,
        phis_start_cf=None,
        x_foundation=None,
        thetas_orientation_cc=None,
        thetas_orientation_cf=None,
        # Clamp info
        fixed_clamp_options=None,
        phis=None,
        # Simsopt info
        fixed_dof_names=None,
        names=None,
        dofs=None,
        **kwargs,
    ):
        # ── Stored for GSONable serialization (structural params) ─────────────────
        # GSONable.as_dict requires every __init__ parameter to be present as
        # self.<param> or self._<param>.
        self.beam_options         = dict(beam_options or {})           # copy before setdefault mutates
        self._fixed_clamp_options = dict(fixed_clamp_options or {'enabled': False})
        self._fixed_dof_names     = fixed_dof_names
        self._names               = names
        # ─────────────────────────────────────────────────────────────────────────

        # ── Stored for GSONable serialization (initial DOF seeds) ─────────────────
        # These are the construction-time seeds for the Optimizable DOF vector.
        # Their actual values are overwritten by the Optimizable dofs object
        # (self._dofs) restored by simsopt on load — these attributes only serve
        # to satisfy GSONable's introspection.
        self._phis_start_cc         = phis_start_cc
        self._phis_end_cc           = phis_end_cc
        self._phis_start_cf         = phis_start_cf
        self._x_foundation          = x_foundation
        self._thetas_orientation_cc = thetas_orientation_cc
        self._thetas_orientation_cf = thetas_orientation_cf
        self._phis                  = phis
        self.kwargs                 = kwargs   # GSONable adds **self.kwargs to the dict
        # ─────────────────────────────────────────────────────────────────────────

        beam_options         = self.beam_options
        fixed_clamp_options  = self._fixed_clamp_options

        n_base = len(base_coils)

        # ── Cross-section presets ─────────────────────────────────────────────
        # The default cross section is solid circle
        cross_section_type = beam_options.get('cross_section_type', 'solid_circle')
        # The default type of attachment is direct 
        # (selecting only exterior nodes inside the beam volume)
        attachment_type = beam_options.get('attachment_type', 'direct')
        cross_section_fn = getattr(cross_section_fns, cross_section_type)
        cross_section_dof_keys = getattr(cross_section_fns, cross_section_type + '_dof_keys')
        cross_section_option_keys = getattr(cross_section_fns, cross_section_type + '_option_keys')

        # ── Attachment function preset ─────────────────────────────────────────
        if attachment_type == 'direct':
            attachment_fn = getattr(cross_section_fns, cross_section_type + '_attachment')
        elif attachment_type == 'wrap':
            attachment_fn = cross_section_fns.wrap_attachment
            cross_section_option_keys += cross_section_fns.wrap_option_keys
        else:
            raise ValueError(
                f"attachment_type={attachment_type!r} not recognized; "
                "must be 'direct' or 'wrap'."
            )

        # ── Load the remaining beam options ───────────────────────────────────
        # Default sigmoid function weight
        beam_options.setdefault('sigmoid_eps', 0.1)  
        missing_beam_options = [k for k in _REQUIRED_BEAM_OPTIONS + cross_section_option_keys if k not in beam_options]
        if missing_beam_options:
            raise ValueError(
                f"beam_options must contain {missing_beam_options}."
            )

        # ── Optional fixed-sphere Winkler clamps ──────────────────────────────
        # Resolved before SupportBeams construction because fixed_clamp_fns is
        # a constructor argument.  The "must have >=1 CF beam" fallback check
        # (disabled-clamp branch) needs n_beam_cf, so it is
        # deferred until after SupportBeams has checked the counts below.
        phis_arr = None
        self._r_clamp = None
        self._sig_eps  = None
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

            self._r_clamp = float(r_clamp)
            self._sig_eps  = float(sigmoid_eps)
            fixed_clamp_fns = self._clamp_fn
        else:
            fixed_clamp_fns = None

        # ── Build the functional SupportBeams ─────────────────────────────────
        # SupportBeams.__init__ is the single owner of the beam-count
        # normalization (including the n_base + 1 stellsym convention); read
        # the final counts back from it instead of recomputing them here.
        beams = SupportBeams(
            nfp=nfp,
            stellsym=stellsym,
            beam_options=beam_options,
            n_base=n_base,
            cross_section_fn=cross_section_fn,
            cross_section_dof_keys=cross_section_dof_keys,
            attachment_fn=attachment_fn,
            fixed_clamp_fns=fixed_clamp_fns,
        )
        n_beam_cc = beams.n_beam_cc
        n_beam_cf = beams.n_beam_cf
        n_groups_cc = beams.n_groups_cc

        if not fixed_clamp_options.get('enabled', False):
            # Each coil must be supported by either fixed clamps (handled
            # above, applies to all coils) or at least one coil-foundation
            # beam.
            unsupported = [i for i in range(n_base) if n_beam_cf[i] < 1]
            if unsupported:
                raise AttributeError(
                    "Each coil must be supported by either fixed clamps or at "
                    "least one coil-foundation beam. Coils with neither: "
                    f"{unsupported}."
                )

        # ── Build initial support_dofs_jax with defaults (ragged per-group) ────
        # Every DOF is a Python list of per-group JAX arrays (a pytree, so
        # ravel_pytree flattens it deterministically even with ragged sizes).
        # CC keys have len(n_beam_cc) == n_groups_cc entries; CF keys have
        # len(n_beam_cf) == n_base entries.
        _phis_end_cc   = _check_ragged_shape(phis_end_cc,           n_beam_cc, 'phis_end_cc')
        _phis_start_cc = _check_ragged_shape(phis_start_cc,         n_beam_cc, 'phis_start_cc')
        _phis_start_cf = _check_ragged_shape(phis_start_cf,         n_beam_cf, 'phis_start_cf')
        _theta_cc      = _check_ragged_shape(thetas_orientation_cc, n_beam_cc, 'thetas_orientation_cc')
        _theta_cf      = _check_ragged_shape(thetas_orientation_cf, n_beam_cf, 'thetas_orientation_cf')
        _x_foundation  = _check_ragged_shape(x_foundation,          n_beam_cf, 'x_foundation', trailing=(3,))

        support_dofs_jax = {
            'phis_end_cc':        _phis_end_cc if _phis_end_cc   is not None else _uniform_list(n_beam_cc, stellsym, True),
            'phis_start_cc':    _phis_start_cc if _phis_start_cc is not None else _uniform_list(n_beam_cc, stellsym),
            'phis_start_cf':    _phis_start_cf if _phis_start_cf is not None else _uniform_list(n_beam_cf),
            'thetas_orientation_cc': _theta_cc if _theta_cc      is not None else _zeros_list(n_beam_cc),
            'thetas_orientation_cf': _theta_cf if _theta_cf      is not None else _zeros_list(n_beam_cf),
            'x_foundation':      _x_foundation if _x_foundation  is not None else _zeros_list(n_beam_cf, trailing=(3,)),
        }
        if phis_arr is not None:
            support_dofs_jax['phis'] = phis_arr

        # Cross-section DOF keys (e.g. radius for solid_circle).
        # Each key becomes a per-group list so every beam can carry its own
        # cross-section parameter as a DOF: entry i < n_base has shape
        # (n_beam_cc[i] + n_beam_cf[i],); with stellsym an extra entry n_base
        # has shape (n_beam_cc[n_base],) for the wrap group (no CF part).
        # Callers may pass a scalar (same value for every beam) or a
        # per-group sequence of arrays.
        _cs_counts = [
            n_beam_cc[i] + (n_beam_cf[i] if i < n_base else 0)
            for i in range(n_groups_cc)
        ]
        if kwargs is None:
            raise AttributeError(
                "The cross section shape requires initial values of "
                f"'{cross_section_dof_keys}' as keyword arguments for "
                "CoilSupportBeams. No keyword arguments are detected."
            )
        for k in cross_section_dof_keys:
            if k not in kwargs:
                raise AttributeError(
                    "The cross section shape requires an initial value of "
                    f"'{k}' as keyword argument for CoilSupportBeams."
                )
            val = kwargs[k]
            if np.ndim(val) == 0:
                support_dofs_jax[k] = [
                    jnp.broadcast_to(
                        jnp.asarray(val, dtype=float), (_cs_counts[i],)
                    )
                    for i in range(n_groups_cc)
                ]
            else:
                seq = list(val)
                if len(seq) != n_groups_cc:
                    raise ValueError(
                        f"Cross-section DOF '{k}' must be a scalar or a "
                        f"length-{n_groups_cc} sequence (one entry per CC "
                        f"group; n_base + 1 when stellsym=True); got length "
                        f"{len(seq)}."
                    )
                support_dofs_jax[k] = [
                    jnp.broadcast_to(
                        jnp.asarray(seq[i], dtype=float), (_cs_counts[i],)
                    )
                    for i in range(n_groups_cc)
                ]

        # ── Compute boolean fixed_mask from fixed_dof_names ───────────────────
        if fixed_dof_names is None:
            fixed_dof_names = list(cross_section_dof_keys) + \
                ['thetas_orientation_cc', 'thetas_orientation_cf']

        fixed_dof_names = list(fixed_dof_names)
        # When *no* coil has CC (resp. CF) beams, those keys are all zero-size
        # leaves anyway; marking them fixed is harmless and documents intent.
        # Per-coil zero counts are handled automatically: an empty leaf
        # contributes no DOFs to the flattened vector.
        if sum(n_beam_cc) == 0:
            fixed_dof_names += ['phis_start_cc', 'phis_end_cc', 'thetas_orientation_cc']
        if sum(n_beam_cf) == 0:
            fixed_dof_names += ['phis_start_cf', 'x_foundation', 'thetas_orientation_cf']

        valid_keys = support_dofs_jax.keys()
        invalid_keys = [k for k in fixed_dof_names if k not in valid_keys]
        if invalid_keys:
            raise ValueError(
                f"fixed_dof_names: {invalid_keys} not valid "
                f"support_dofs key(s). Valid keys: {valid_keys}"
            )
        # Build the boolean fixed-mask over the same (possibly ragged) pytree
        # structure as support_dofs_jax so it stays bit-aligned after ravel.
        probe = {
            k: jax.tree_util.tree_map(
                lambda leaf, kk=k: np.full(np.shape(leaf), kk in fixed_dof_names, dtype=bool),
                v,
            )
            for k, v in support_dofs_jax.items()
        }
        fixed_mask, _ = ravel_pytree(probe)

        # ── Summary ───────────────────────────────────────────────────────────
        def _shapes(v):
            if isinstance(v, list):
                return [tuple(np.shape(leaf)) for leaf in v]
            return tuple(np.shape(v))

        print('Beam network initialized.')
        print('Optimizing:')
        for k in valid_keys:
            if k not in fixed_dof_names:
                print('   ', k, '- per-coil shapes:', _shapes(support_dofs_jax[k]))
        print('Fixing:')
        for k in fixed_dof_names:
            print('   ', k)
        print('Total # dofs:', len(fixed_mask) - int(np.sum(fixed_mask)))

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

    def _clamp_fn(self, surface_pts, curve_jax, dofs_slice):
        """Winkler weight for one coil; identical body to CoilSupportFixed._clamp_fn."""
        return CoilSupportFixed._clamp_fn(self, surface_pts, curve_jax, dofs_slice)
