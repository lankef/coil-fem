"""Beam-network simsopt coil support models.

:class:`CoilSupportBeams` and :class:`CoilSupportBeamsSorted` wrap
:class:`~coil_fem.coupling.SupportBeams` as simsopt Optimizables.
"""

from __future__ import annotations

import warnings

import numpy as np

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map

from ..presets import cross_section_fns
from ..utils import estimate_k
from ..geo import CurveXYZFourierJAX
from ..coupling import SupportBeams
from .coil_support import (
    CoilSupport,
    _ANGLE_UNIT_KEYS,
    _DPHI_TO_PHI,
    _PHI_TO_DPHI,
    _SortedDphisMixin,
    _apply_sorted_dphi_bounds,
    _broadcast_phis,
    _cumsum_last,
    _cumsum_last_vjp,
    _diff_last,
    _encode_dphis,
    _fold_first_dphis,
    _generate_k_clamp,
    _tree_cumsum_last,
)
from .coil_support_fixed import CoilSupportFixed


_REQUIRED_BEAM_OPTIONS = (
    'n_beam_cc',
    'n_beam_cf',
    'E',
    'nu',
    'cross_section_type',
    'attachment_type',
)
# Optional; auto-generated from beam stiffness when omitted.
_OPTIONAL_BEAM_OPTIONS = (
    'k_attachment',
    'eps_attachment',
)


# ============================================================================
# CoilSupportBeams construction helpers (used only during __init__)
# ============================================================================

def _uniform_list(counts, cc_stellsym=False, cc_end=False):
    """Build a per-group list of uniformly spaced initial phi values.

    Used during :class:`CoilSupportBeams` construction to initialise
    ``phis_start_cc``, ``phis_end_cc``, and ``phis_start_cf`` when the caller
    does not supply explicit values.

    When ``cc_stellsym=True``, the last two entries (the stellsym wrap groups)
    use a restricted half-period range to avoid beam overlap after the
    stellarator reflection: starts in ``[0, 0.5)``, ends as the elementwise
    complement ``1 - starts`` (descending in ``(0.5, 1]``) so beam ``j``
    pairs as ``phi_end[j] = 1 - phi_start[j]``.
    """
    out = []
    for i in range(len(counts)):
        c = counts[i]
        if c == 0:
            out.append(jnp.array([]))
            continue
        wrap = cc_stellsym and i >= len(counts) - 2
        if wrap:
            # Same linspace for starts and ends so pairing is exact.
            phi = jnp.linspace(0.0, 0.5, c, endpoint=False) + 0.25 / c
            if cc_end:
                phi = 1.0 - phi  # descending; pairs index-wise with starts
        else:
            phi = jnp.linspace(0.0, 1.0, c, endpoint=False) + 0.5 / c
        out.append(phi)
    return out


# ============================================================================
# Stellsym wrap-end dphis codec (backward walk from phi = 1)
# ============================================================================

def _wrap_end_groups(n_groups, stellsym):
    """Indices of CC groups whose ends walk backward under stellsym.

    Returns the empty set when ``stellsym`` is false or there are fewer than
    two groups.  Otherwise returns ``{n_groups - 2, n_groups - 1}`` — the
    ``flip_half`` and ``flip`` wrap groups.
    """
    if not stellsym or n_groups < 2:
        return frozenset()
    return frozenset({n_groups - 2, n_groups - 1})


def _decode_end_cc(d_list, stellsym):
    """Decode ``dphis_end_cc`` → ``phis_end_cc`` (wrap groups: ``1 - cumsum``)."""
    wrap = _wrap_end_groups(len(d_list), stellsym)
    out = []
    for g, d in enumerate(d_list):
        d = jnp.asarray(d, dtype=float)
        if g in wrap:
            out.append(1.0 - _cumsum_last(d))
        else:
            out.append(_cumsum_last(d))
    return out


def _encode_end_cc(phi_list, stellsym):
    """Encode ``phis_end_cc`` → ``dphis_end_cc`` (wrap: ``diff(1 - phi)``)."""
    wrap = _wrap_end_groups(len(phi_list), stellsym)
    out = []
    for g, phi in enumerate(phi_list):
        phi = jnp.asarray(phi, dtype=float)
        if g in wrap:
            # diff_last(1 - phi), not -diff_last(phi): they differ at j=0.
            out.append(_diff_last(1.0 - phi))
        else:
            out.append(_diff_last(phi))
    return out


def _vjp_end_cc(g_list, stellsym):
    """VJP of :func:`_decode_end_cc` (wrap groups: negated reverse-cumsum)."""
    wrap = _wrap_end_groups(len(g_list), stellsym)
    out = []
    for g, g_phi in enumerate(g_list):
        g_phi = jnp.asarray(g_phi, dtype=float)
        if g in wrap:
            out.append(-_cumsum_last_vjp(g_phi))
        else:
            out.append(_cumsum_last_vjp(g_phi))
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
            translational and rotational spring coupling.
        eps_sigmoid : float
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
        Initial cross-section roll angles for CC beams (fraction of a turn
        in ``[0, 1]``), one entry per CC group with entry ``g`` of shape
        ``(n_beam_cc[g],)``.
    thetas_orientation_cf : sequence of array-like or None
        Initial cross-section roll angles for CF beams (fraction of a turn
        in ``[0, 1]``), a length-``n_base`` sequence with entry ``i`` of
        shape ``(n_beam_cf[i],)``.
    fixed_clamp_options : dict
        Optional additional fixed-sphere Winkler clamps on the coil surface.
        Set ``{'enabled': True, 'k_clamp': ..., 'r_clamp': ..., 'n_clamp': ...}``
        to enable; ``{'enabled': False}`` (default) disables.
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
        beam_options=(),
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
        # ── Stored for GSONable serialization (initial DOF seeds) ─────────────────
        # These are the construction-time seeds for the Optimizable DOF vector.
        # Their actual values are overwritten by the Optimizable dofs object
        # (self._dofs) restored by simsopt on load — these attributes only serve
        # to satisfy GSONable's introspection.
        # GSONable.as_dict requires every __init__ parameter to be present as
        # self.<param> or self._<param>.
        self._fixed_dof_names       = fixed_dof_names
        self._names                 = names
        self._phis_start_cc         = phis_start_cc
        self._phis_end_cc           = phis_end_cc
        self._phis_start_cf         = phis_start_cf
        self._x_foundation          = x_foundation
        self._thetas_orientation_cc = thetas_orientation_cc
        self._thetas_orientation_cf = thetas_orientation_cf
        self._phis                  = phis
        self.kwargs                 = kwargs   # GSONable adds **self.kwargs to the dict
        # ─────────────────────────────────────────────────────────────────────────

        # Resolve options into local copies; caller-owned dicts are never mutated.
        # Assigned to self.beam_options / self._fixed_clamp_options once resolved
        # so GSONable serialises the filled-in values.
        beam_options = {'eps_sigmoid': 0.1, **dict(beam_options or {})}
        fixed_clamp_options = dict(fixed_clamp_options or {'enabled': False})

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
        beam_option_keys_req = _REQUIRED_BEAM_OPTIONS + cross_section_option_keys
        beam_option_keys_allowed = (
            beam_option_keys_req + _OPTIONAL_BEAM_OPTIONS
        )
        missing_beam_options = [k for k in beam_option_keys_req if k not in beam_options]
        unrecognized_beam_options = [
            k for k in beam_options if k not in beam_option_keys_allowed
        ]
        if missing_beam_options:
            raise ValueError(
                f"Missing keys in beam_options: {missing_beam_options}."
            )
        if unrecognized_beam_options:
            warnings.warn(
                f"Unrecognized keys in beam_options: {unrecognized_beam_options}."
            )

        # Calculating beam options
        if 'k_attachment' not in beam_options:
            eps_attachment = beam_options.get('eps_attachment', 1e-3)
            E_beams = beam_options['E']
            centers_zero = CurveXYZFourierJAX.from_simsopt(
                base_coils[0].curve
            ).curve_center()
            estimated_R = jnp.sqrt(centers_zero[0]**2 + centers_zero[1]**2)
            displacements = estimated_R * jnp.pi * 2 / nfp / len(base_coils)
            if stellsym:
                displacements = displacements / 2
            k_attachment = estimate_k(
                L=np.mean(displacements),
                E=E_beams,
                eps=eps_attachment,
            )
            beam_options = {**beam_options, 'k_attachment': float(k_attachment)}
            print(
                "k_attachment is not provided. Based on the coil's stiffness, "
                f"the auto-generated value is {k_attachment:.4e} N/m3."
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
            # k_clamp, the Robin/Winkler BC, can be auto-generated using the  
            # stiffness of the coil body.
            k_clamp = _generate_k_clamp(base_coils, fixed_clamp_options)
            try:
                r_clamp = fixed_clamp_options['r_clamp']
                n_clamp = fixed_clamp_options['n_clamp']
            except KeyError:
                raise KeyError(
                    "fixed_clamp_options must contain 'r_clamp' and 'n_clamp'."
                )
            eps_sigmoid = fixed_clamp_options.get('eps_sigmoid', 0.1)

            phis_arr = _broadcast_phis(phis, n_base, n_clamp)

            self._r_clamp = float(r_clamp)
            self._sig_eps  = float(eps_sigmoid)
            fixed_clamp_fns = self._clamp_fn
            # Keep k_clamp in the stored options so SupportBeams / GSONable
            # see the same dict that was validated here.
            fixed_clamp_options = {
                **fixed_clamp_options, 'k_clamp': float(k_clamp),
            }
        else:
            fixed_clamp_fns = None

        self.beam_options = beam_options
        self._fixed_clamp_options = fixed_clamp_options

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
            fixed_clamp_options=fixed_clamp_options,
        )
        n_beam_cc = beams.n_beam_cc
        n_beam_cf = beams.n_beam_cf
        n_groups_cc = beams.n_groups_cc

        # if not fixed_clamp_options.get('enabled', False):
        #     # Each coil must be supported by either fixed clamps (handled
        #     # above, applies to all coils) or at least one coil-foundation
        #     # beam.
        #     unsupported = [i for i in range(n_base) if n_beam_cf[i] < 1]
        #     if unsupported:
        #         raise AttributeError(
        #             "Each coil must be supported by either fixed clamps or at "
        #             "least one coil-foundation beam. Coils with neither: "
        #             f"{unsupported}."
        #         )

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
        # By design, kwarg are the initial values for cross section 
        # dofs.
        for k in kwargs:
            if k not in cross_section_dof_keys:
                warnings.warn(
                    f"Unrecognized key word argument: {k}. Key word arguments "
                    "that are not CoilSupportBeam parameters are reserved"
                    "for initial values for cross section dofs, such as beam widths."
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

        support_dofs_jax, fixed_dof_names = self._encode_angle_dofs(
            support_dofs_jax, fixed_dof_names,
        )

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
            k: tree_map(
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

        unit_interval_keys = tuple(
            k for k in support_dofs_jax if k in _ANGLE_UNIT_KEYS
        ) + ('thetas_orientation_cc', 'thetas_orientation_cf')
        lb, ub = self._make_bounds(
            support_dofs_jax,
            unit_interval_keys=unit_interval_keys,
            nonnegative_keys=tuple(cross_section_dof_keys),
        )
        lb, ub = _apply_sorted_dphi_bounds(
            lb, ub, support_dofs_jax, nfp, stellsym,
        )

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
            lower_bounds=lb,
            upper_bounds=ub,
            dofs=dofs,
        )

    def _encode_angle_dofs(self, support_dofs_jax, fixed_dof_names):
        """Hook for Sorted subclasses to store ``dphis*`` instead of ``phis*``."""
        return support_dofs_jax, fixed_dof_names

    def _clamp_fn(self, surface_pts, curve_jax, dofs_slice):
        """Winkler weight for one coil; identical body to CoilSupportFixed._clamp_fn."""
        return CoilSupportFixed._clamp_fn(self, surface_pts, curve_jax, dofs_slice)


class CoilSupportBeamsSorted(_SortedDphisMixin, CoilSupportBeams):
    """:class:`CoilSupportBeams` with incremental ``dphis*`` angle DOFs.

    Simsopt stores ``dphis_start_cc``, ``dphis_end_cc``, ``dphis_start_cf``
    (and optional clamp ``dphis``) with absolute angles recovered by
    ``cumsum`` along the last axis — except for the two stellsym wrap
    groups, where ``phis_end_cc`` is recovered as ``1 - cumsum(dphis_end_cc)``
    (a positive step *backward* from ``phi = 1``).  The first increment of
    ``dphis_start_cc``, ``dphis_end_cc``, and ``dphis_start_cf`` is boxed
    to ``[-0.5, 0.5]`` (later increments stay in ``[0, 1]``) and default
    first values are folded into that interval.  :attr:`support_dofs`
    exposes ``phis*`` for the FEM; :meth:`flatten_grad` applies the
    matching VJP.

    Parameters
    ----------
    base_coils, nfp, stellsym, beam_options, x_foundation,
    thetas_orientation_cc, thetas_orientation_cf, fixed_clamp_options,
    fixed_dof_names, names, dofs, **kwargs
        Same as :class:`CoilSupportBeams`, except angle seeds use ``dphis*``
        (see below).  ``fixed_dof_names`` may use either ``phis*`` or
        ``dphis*`` key spellings.
    dphis_start_cc, dphis_end_cc, dphis_start_cf : sequence of array-like or None
        Initial increments for CC/CF attachment angles (same ragged shapes as
        the corresponding ``phis_*`` arguments of :class:`CoilSupportBeams`).
        For stellsym wrap groups, ``dphis_end_cc`` steps backward from 1.
    dphis : array-like or None
        Initial increments for optional fixed-sphere clamps.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        beam_options=(),
        dphis_start_cc=None,
        dphis_end_cc=None,
        dphis_start_cf=None,
        x_foundation=None,
        thetas_orientation_cc=None,
        thetas_orientation_cf=None,
        fixed_clamp_options=None,
        dphis=None,
        fixed_dof_names=None,
        names=None,
        dofs=None,
        **kwargs,
    ):
        # Must precede super().__init__: _encode_angle_dofs runs during
        # CoilSupportBeams.__init__ and needs these flags.
        self._sorted_stellsym = bool(stellsym)
        self._sorted_nfp = int(nfp)
        self._dphis_start_cc = dphis_start_cc
        self._dphis_end_cc = dphis_end_cc
        self._dphis_start_cf = dphis_start_cf
        self._dphis = dphis
        if fixed_dof_names is not None:
            fixed_dof_names = [_DPHI_TO_PHI.get(k, k) for k in fixed_dof_names]
        super().__init__(
            base_coils,
            nfp,
            stellsym,
            beam_options=beam_options,
            phis_start_cc=(
                _tree_cumsum_last(dphis_start_cc)
                if dphis_start_cc is not None else None
            ),
            phis_end_cc=(
                _decode_end_cc(list(dphis_end_cc), stellsym)
                if dphis_end_cc is not None else None
            ),
            phis_start_cf=(
                _tree_cumsum_last(dphis_start_cf)
                if dphis_start_cf is not None else None
            ),
            x_foundation=x_foundation,
            thetas_orientation_cc=thetas_orientation_cc,
            thetas_orientation_cf=thetas_orientation_cf,
            fixed_clamp_options=fixed_clamp_options,
            phis=(
                _cumsum_last(jnp.asarray(dphis, dtype=float))
                if dphis is not None else None
            ),
            fixed_dof_names=fixed_dof_names,
            names=names,
            dofs=dofs,
            **kwargs,
        )

    def _encode_angle_dofs(self, support_dofs_jax, fixed_dof_names):
        encoded = _encode_dphis(support_dofs_jax)
        if 'phis_end_cc' in support_dofs_jax:
            encoded['dphis_end_cc'] = _encode_end_cc(
                support_dofs_jax['phis_end_cc'], self._sorted_stellsym,
            )
        encoded = _fold_first_dphis(
            encoded, self._sorted_nfp, self._sorted_stellsym,
        )
        renamed = [_PHI_TO_DPHI.get(k, k) for k in fixed_dof_names]
        return encoded, renamed
