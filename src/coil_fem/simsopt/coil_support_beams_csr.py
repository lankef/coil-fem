"""Simsopt wrapper for :class:`~coil_fem.coupling.SupportBeamsCSR`.

:class:`CoilSupportBeamsCSR` and :class:`CoilSupportBeamsCSRSorted` wrap
:class:`~coil_fem.coupling.SupportBeamsCSR` as simsopt Optimizables.
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
from ..coupling import SupportBeamsCSR
from .coil_support import (
    CoilSupport,
    _ANGLE_UNIT_KEYS,
    _DPHI_TO_PHI,
    _PHI_TO_DPHI,
    _SortedDphisMixin,
    _broadcast_phis,
    _cumsum_last,
    _encode_dphis,
    _generate_k_clamp,
    _tree_cumsum_last,
)
from .coil_support_fixed import CoilSupportFixed
from .coil_support_beams import (
    _REQUIRED_BEAM_OPTIONS,
    _OPTIONAL_BEAM_OPTIONS,
    _uniform_list,
    _zeros_list,
    _check_ragged_shape,
    _decode_end_cc,
    _encode_end_cc,
)


_REQUIRED_CSR_OPTIONS = ('order', 'w1', 'w2', 'n_phi', 'E', 'nu')
_OPTIONAL_CSR_OPTIONS = ('n_grid_1', 'n_grid_2', 'mesh_type', 'n_quad')


def _coil_center_phi_list(base_coils, n_beam_cr):
    """Per-coil ``phis_end_cr`` seed at each coil's cylindrical angle.

    Entry ``i`` has shape ``(n_beam_cr[i],)`` with all values equal to
    ``atan2(y_c, x_c) / (2π) mod 1`` where ``(x_c, y_c, z_c)`` is the
    coil's ``curve_center``.
    """
    out = []
    for i, coil in enumerate(base_coils):
        c = CurveXYZFourierJAX.from_simsopt(coil.curve).curve_center()
        phi_i = (jnp.arctan2(c[1], c[0]) / (2.0 * jnp.pi)) % 1.0
        out.append(jnp.full((int(n_beam_cr[i]),), phi_i))
    return out


def _v_end_cr_linspace(n_beam_cr):
    """Per-coil ``v_end_cr`` from -1 to 1 (``n=1`` → 0)."""
    out = []
    for n in n_beam_cr:
        n = int(n)
        if n == 0:
            out.append(jnp.zeros(0))
        elif n == 1:
            out.append(jnp.zeros(1))
        else:
            out.append(jnp.linspace(-1.0, 1.0, n))
    return out


def _min_R_phi_window_list(base_coils, n_beam_cr, width=0.25):
    """``phis_start_cr`` in a width-``width`` window around min cylindrical R.

    For each coil, sample ``gamma`` on its quadpoints, take
    ``phi0 = argmin sqrt(x^2+y^2)``, place ``n`` points with
    ``linspace(phi0 - width/2, phi0 + width/2, n)``, then ``mod 1`` and
    sort ascending so Sorted ``dphis_start_cr`` stay non-negative.

    ponytail: sorting after ``% 1`` can reorder beam indices when the window
    wraps across 0; upgrade path is a wrap-aware Sorted codec.
    """
    half = 0.5 * float(width)
    out = []
    for i, coil in enumerate(base_coils):
        n = int(n_beam_cr[i])
        if n == 0:
            out.append(jnp.zeros(0))
            continue
        curve = CurveXYZFourierJAX.from_simsopt(coil.curve)
        gamma = curve.gamma()
        R = jnp.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
        phi0 = curve.quadpoints[jnp.argmin(R)]
        if n == 1:
            out.append(jnp.array([phi0 % 1.0]))
        else:
            phis = jnp.linspace(phi0 - half, phi0 + half, n) % 1.0
            out.append(jnp.sort(phis))
    return out


class CoilSupportBeamsCSR(CoilSupport):
    """Simsopt-optimizable CSR beam-network coil support.

    Wraps :class:`~coil_fem.coupling.SupportBeamsCSR`.  In addition to the
    CC/CF DOFs of :class:`CoilSupportBeams`, exposes CR attachment angles
    (``phis_start_cr``, ``phis_end_cr``), the cross-section offset
    ``v_end_cr``, CR orientations, and ``csr_curve_dofs``.

    Parameters
    ----------
    base_coils, nfp, stellsym
        Same as :class:`CoilSupportBeams`.
    beam_options : dict
        Same as :class:`CoilSupportBeams`, plus required ``n_beam_cr``.
    csr_options : dict
        Forwarded to :class:`~coil_fem.coupling.SupportBeamsCSR`.  Required
        keys: ``order``, ``w1``, ``w2``, ``n_phi``, ``E``, ``nu``.
    problem_options : dict
        Same dict passed to :class:`~coil_fem.CoilFEM` (controls
        ``gpu_assembly`` for the CSR pipeline).
    phis_start_cc, phis_end_cc, phis_start_cf, x_foundation,
    thetas_orientation_cc, thetas_orientation_cf
        Same as :class:`CoilSupportBeams`.
    phis_start_cr, phis_end_cr, v_end_cr, thetas_orientation_cr
        Per-coil ragged CR DOFs; entry ``i`` has shape ``(n_beam_cr[i],)``.
    csr_curve_dofs : array-like or None
        Initial CSR :class:`~coil_fem.geo.CurveRZFourierJAX` DOFs.  Default
        is a unit circle (``rc_0 = 1``).
    fixed_clamp_options, phis, fixed_dof_names, names, dofs, **kwargs
        Same as :class:`CoilSupportBeams`.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        beam_options=(),
        csr_options=(),
        problem_options=None,
        phis_start_cc=None,
        phis_end_cc=None,
        phis_start_cf=None,
        phis_start_cr=None,
        phis_end_cr=None,
        v_end_cr=None,
        x_foundation=None,
        thetas_orientation_cc=None,
        thetas_orientation_cf=None,
        thetas_orientation_cr=None,
        csr_curve_dofs=None,
        fixed_clamp_options=None,
        phis=None,
        fixed_dof_names=None,
        names=None,
        dofs=None,
        **kwargs,
    ):
        self._fixed_dof_names = fixed_dof_names
        self._names = names
        self._phis_start_cc = phis_start_cc
        self._phis_end_cc = phis_end_cc
        self._phis_start_cf = phis_start_cf
        self._phis_start_cr = phis_start_cr
        self._phis_end_cr = phis_end_cr
        self._v_end_cr = v_end_cr
        self._x_foundation = x_foundation
        self._thetas_orientation_cc = thetas_orientation_cc
        self._thetas_orientation_cf = thetas_orientation_cf
        self._thetas_orientation_cr = thetas_orientation_cr
        self._csr_curve_dofs = csr_curve_dofs
        self._phis = phis
        self.kwargs = kwargs

        beam_options = {'eps_sigmoid': 0.1, **dict(beam_options or {})}
        csr_options = dict(csr_options or {})
        problem_options = dict(problem_options or {})
        fixed_clamp_options = dict(fixed_clamp_options or {'enabled': False})
        n_base = len(base_coils)

        if 'n_beam_cr' not in beam_options:
            raise ValueError("beam_options must contain 'n_beam_cr'.")
        missing_csr = [k for k in _REQUIRED_CSR_OPTIONS if k not in csr_options]
        if missing_csr:
            raise ValueError(f"csr_options missing required keys: {missing_csr}.")
        unrecognized_csr = [
            k for k in csr_options
            if k not in _REQUIRED_CSR_OPTIONS + _OPTIONAL_CSR_OPTIONS
        ]
        if unrecognized_csr:
            warnings.warn(
                f"Unrecognized keys in csr_options: {unrecognized_csr}."
            )

        cross_section_type = beam_options.get('cross_section_type', 'solid_circle')
        attachment_type = beam_options.get('attachment_type', 'direct')
        cross_section_fn = getattr(cross_section_fns, cross_section_type)
        cross_section_dof_keys = getattr(
            cross_section_fns, cross_section_type + '_dof_keys',
        )
        cross_section_option_keys = getattr(
            cross_section_fns, cross_section_type + '_option_keys',
        )

        if attachment_type == 'direct':
            attachment_fn = getattr(
                cross_section_fns, cross_section_type + '_attachment',
            )
        elif attachment_type == 'wrap':
            attachment_fn = cross_section_fns.wrap_attachment
            cross_section_option_keys += cross_section_fns.wrap_option_keys
        else:
            raise ValueError(
                f"attachment_type={attachment_type!r} not recognized; "
                "must be 'direct' or 'wrap'."
            )

        beam_option_keys_req = (
            _REQUIRED_BEAM_OPTIONS + ('n_beam_cr',) + cross_section_option_keys
        )
        beam_option_keys_allowed = (
            beam_option_keys_req + _OPTIONAL_BEAM_OPTIONS
        )
        missing_beam = [k for k in beam_option_keys_req if k not in beam_options]
        if missing_beam:
            raise ValueError(f"Missing keys in beam_options: {missing_beam}.")
        unrecognized_beam = [
            k for k in beam_options if k not in beam_option_keys_allowed
        ]
        if unrecognized_beam:
            warnings.warn(
                f"Unrecognized keys in beam_options: {unrecognized_beam}."
            )

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

        phis_arr = None
        self._r_clamp = None
        self._sig_eps = None
        if fixed_clamp_options.get('enabled', False):
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
            self._sig_eps = float(eps_sigmoid)
            fixed_clamp_fns = self._clamp_fn
            fixed_clamp_options = {
                **fixed_clamp_options, 'k_clamp': float(k_clamp),
            }
        else:
            fixed_clamp_fns = None

        self.beam_options = beam_options
        self.csr_options = csr_options
        self.problem_options = problem_options
        self._fixed_clamp_options = fixed_clamp_options

        beams = SupportBeamsCSR(
            nfp=nfp,
            stellsym=stellsym,
            beam_options=beam_options,
            n_base=n_base,
            cross_section_fn=cross_section_fn,
            cross_section_dof_keys=cross_section_dof_keys,
            attachment_fn=attachment_fn,
            csr_options=csr_options,
            problem_options=problem_options,
            fixed_clamp_fns=fixed_clamp_fns,
            fixed_clamp_options=fixed_clamp_options,
        )
        n_beam_cc = beams.n_beam_cc
        n_beam_cf = beams.n_beam_cf
        n_beam_cr = beams.n_beam_cr
        n_groups_cc = beams.n_groups_cc

        _phis_end_cc = _check_ragged_shape(
            phis_end_cc, n_beam_cc, 'phis_end_cc',
        )
        _phis_start_cc = _check_ragged_shape(
            phis_start_cc, n_beam_cc, 'phis_start_cc',
        )
        _phis_start_cf = _check_ragged_shape(
            phis_start_cf, n_beam_cf, 'phis_start_cf',
        )
        _phis_start_cr = _check_ragged_shape(
            phis_start_cr, n_beam_cr, 'phis_start_cr',
        )
        _phis_end_cr = _check_ragged_shape(
            phis_end_cr, n_beam_cr, 'phis_end_cr',
        )
        _v_end_cr = _check_ragged_shape(v_end_cr, n_beam_cr, 'v_end_cr')
        _theta_cc = _check_ragged_shape(
            thetas_orientation_cc, n_beam_cc, 'thetas_orientation_cc',
        )
        _theta_cf = _check_ragged_shape(
            thetas_orientation_cf, n_beam_cf, 'thetas_orientation_cf',
        )
        _theta_cr = _check_ragged_shape(
            thetas_orientation_cr, n_beam_cr, 'thetas_orientation_cr',
        )
        _x_foundation = _check_ragged_shape(
            x_foundation, n_beam_cf, 'x_foundation', trailing=(3,),
        )

        order = int(csr_options['order'])
        if stellsym:
            n_csr_dofs = 2 * order + 1
        else:
            n_csr_dofs = 4 * order + 2
        if csr_curve_dofs is None:
            csr_dofs_arr = jnp.zeros(n_csr_dofs)
            csr_dofs_arr = csr_dofs_arr.at[0].set(1.0)
        else:
            csr_dofs_arr = jnp.asarray(csr_curve_dofs, dtype=float)
            if csr_dofs_arr.shape != (n_csr_dofs,):
                raise ValueError(
                    f"csr_curve_dofs must have shape ({n_csr_dofs},); "
                    f"got {csr_dofs_arr.shape}."
                )

        support_dofs_jax = {
            'phis_end_cc': (
                _phis_end_cc if _phis_end_cc is not None
                else _uniform_list(n_beam_cc, stellsym, True)
            ),
            'phis_start_cc': (
                _phis_start_cc if _phis_start_cc is not None
                else _uniform_list(n_beam_cc, stellsym)
            ),
            'phis_start_cf': (
                _phis_start_cf if _phis_start_cf is not None
                else _uniform_list(n_beam_cf)
            ),
            'phis_start_cr': (
                _phis_start_cr if _phis_start_cr is not None
                else _min_R_phi_window_list(base_coils, n_beam_cr)
            ),
            'phis_end_cr': (
                _phis_end_cr if _phis_end_cr is not None
                else _coil_center_phi_list(base_coils, n_beam_cr)
            ),
            'v_end_cr': (
                _v_end_cr if _v_end_cr is not None
                else _v_end_cr_linspace(n_beam_cr)
            ),
            'thetas_orientation_cc': (
                _theta_cc if _theta_cc is not None
                else _zeros_list(n_beam_cc)
            ),
            'thetas_orientation_cf': (
                _theta_cf if _theta_cf is not None
                else _zeros_list(n_beam_cf)
            ),
            'thetas_orientation_cr': (
                _theta_cr if _theta_cr is not None
                else _zeros_list(n_beam_cr)
            ),
            'x_foundation': (
                _x_foundation if _x_foundation is not None
                else _zeros_list(n_beam_cf, trailing=(3,))
            ),
            'csr_curve_dofs': csr_dofs_arr,
        }
        if phis_arr is not None:
            support_dofs_jax['phis'] = phis_arr

        # Cross-section DOFs: CC + CF + CR per base coil; wrap group CC only.
        _cs_counts = [
            n_beam_cc[i]
            + (n_beam_cf[i] if i < n_base else 0)
            + (n_beam_cr[i] if i < n_base else 0)
            for i in range(n_groups_cc)
        ]
        for k in kwargs:
            if k not in cross_section_dof_keys:
                warnings.warn(
                    f"Unrecognized key word argument: {k}. Keyword arguments "
                    "that are not CoilSupportBeamsCSR parameters are reserved "
                    "for initial values for cross section dofs."
                )
        for k in cross_section_dof_keys:
            if k not in kwargs:
                raise AttributeError(
                    "The cross section shape requires an initial value of "
                    f"'{k}' as keyword argument for CoilSupportBeamsCSR."
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
                        f"length-{n_groups_cc} sequence; got length {len(seq)}."
                    )
                support_dofs_jax[k] = [
                    jnp.broadcast_to(
                        jnp.asarray(seq[i], dtype=float), (_cs_counts[i],)
                    )
                    for i in range(n_groups_cc)
                ]

        if fixed_dof_names is None:
            fixed_dof_names = list(cross_section_dof_keys) + [
                'thetas_orientation_cc',
                'thetas_orientation_cf',
                'thetas_orientation_cr',
                'csr_curve_dofs',
            ]
        fixed_dof_names = list(fixed_dof_names)

        if sum(n_beam_cc) == 0:
            fixed_dof_names += [
                'phis_start_cc', 'phis_end_cc', 'thetas_orientation_cc',
            ]
        if sum(n_beam_cf) == 0:
            fixed_dof_names += [
                'phis_start_cf', 'x_foundation', 'thetas_orientation_cf',
            ]
        if sum(n_beam_cr) == 0:
            fixed_dof_names += [
                'phis_start_cr', 'phis_end_cr', 'v_end_cr',
                'thetas_orientation_cr',
            ]

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
        probe = {
            k: tree_map(
                lambda leaf, kk=k: np.full(
                    np.shape(leaf), kk in fixed_dof_names, dtype=bool,
                ),
                v,
            )
            for k, v in support_dofs_jax.items()
        }
        fixed_mask, _ = ravel_pytree(probe)

        def _shapes(v):
            if isinstance(v, list):
                return [tuple(np.shape(leaf)) for leaf in v]
            return tuple(np.shape(v))

        print('CSR beam network initialized.')
        print('Optimizing:')
        for k in valid_keys:
            if k not in fixed_dof_names:
                print('   ', k, '- shapes:', _shapes(support_dofs_jax[k]))
        print('Fixing:')
        for k in fixed_dof_names:
            print('   ', k)
        print('Total # dofs:', len(fixed_mask) - int(np.sum(fixed_mask)))

        unit_interval_keys = tuple(
            k for k in support_dofs_jax if k in _ANGLE_UNIT_KEYS
        ) + (
            'thetas_orientation_cc',
            'thetas_orientation_cf',
            'thetas_orientation_cr',
        )
        lb, ub = self._make_bounds(
            support_dofs_jax,
            unit_interval_keys=unit_interval_keys,
            nonnegative_keys=tuple(cross_section_dof_keys),
        )
        # v_end_cr ∈ [-1, 1]
        v_probe = {
            k: tree_map(
                lambda leaf, kk=k: np.full(
                    np.shape(leaf), kk == 'v_end_cr', dtype=bool,
                ),
                v,
            )
            for k, v in support_dofs_jax.items()
        }
        v_mask, _ = ravel_pytree(v_probe)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        lb = np.where(v_mask, -1.0, lb)
        ub = np.where(v_mask, 1.0, ub)

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
        """Winkler weight for one coil; identical body to CoilSupportFixed."""
        return CoilSupportFixed._clamp_fn(self, surface_pts, curve_jax, dofs_slice)


class CoilSupportBeamsCSRSorted(_SortedDphisMixin, CoilSupportBeamsCSR):
    """:class:`CoilSupportBeamsCSR` with incremental ``dphis*`` angle DOFs.

    Simsopt stores ``dphis_start_cc``, ``dphis_end_cc``, ``dphis_start_cf``,
    ``dphis_start_cr``, ``dphis_end_cr`` (and optional clamp ``dphis``) with
    absolute angles recovered by ``cumsum`` along the last axis — except for
    the two stellsym wrap groups, where ``phis_end_cc`` is recovered as
    ``1 - cumsum(dphis_end_cc)``.  CR ends use a plain cumsum (no wrap-back).
    :attr:`support_dofs` exposes ``phis*`` for the FEM; :meth:`flatten_grad`
    applies the matching VJP.

    Parameters
    ----------
    base_coils, nfp, stellsym, beam_options, csr_options, problem_options,
    v_end_cr, x_foundation, thetas_orientation_cc, thetas_orientation_cf,
    thetas_orientation_cr, csr_curve_dofs, fixed_clamp_options,
    fixed_dof_names, names, dofs, **kwargs
        Same as :class:`CoilSupportBeamsCSR`, except angle seeds use ``dphis*``
        (see below).  ``fixed_dof_names`` may use either ``phis*`` or
        ``dphis*`` key spellings.
    dphis_start_cc, dphis_end_cc, dphis_start_cf,
    dphis_start_cr, dphis_end_cr : sequence of array-like or None
        Initial increments for CC/CF/CR attachment angles (same ragged shapes
        as the corresponding ``phis_*`` arguments of
        :class:`CoilSupportBeamsCSR`).  For stellsym wrap groups,
        ``dphis_end_cc`` steps backward from 1.
    dphis : array-like or None
        Initial increments for optional fixed-sphere clamps.
    """

    def __init__(
        self,
        base_coils: list,
        nfp: int,
        stellsym: bool,
        beam_options=(),
        csr_options=(),
        problem_options=None,
        dphis_start_cc=None,
        dphis_end_cc=None,
        dphis_start_cf=None,
        dphis_start_cr=None,
        dphis_end_cr=None,
        v_end_cr=None,
        x_foundation=None,
        thetas_orientation_cc=None,
        thetas_orientation_cf=None,
        thetas_orientation_cr=None,
        csr_curve_dofs=None,
        fixed_clamp_options=None,
        dphis=None,
        fixed_dof_names=None,
        names=None,
        dofs=None,
        **kwargs,
    ):
        # Must precede super().__init__: _encode_angle_dofs runs during
        # CoilSupportBeamsCSR.__init__ and needs this flag.
        self._sorted_stellsym = bool(stellsym)
        self._dphis_start_cc = dphis_start_cc
        self._dphis_end_cc = dphis_end_cc
        self._dphis_start_cf = dphis_start_cf
        self._dphis_start_cr = dphis_start_cr
        self._dphis_end_cr = dphis_end_cr
        self._dphis = dphis
        if fixed_dof_names is not None:
            fixed_dof_names = [_DPHI_TO_PHI.get(k, k) for k in fixed_dof_names]
        super().__init__(
            base_coils,
            nfp,
            stellsym,
            beam_options=beam_options,
            csr_options=csr_options,
            problem_options=problem_options,
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
            phis_start_cr=(
                _tree_cumsum_last(dphis_start_cr)
                if dphis_start_cr is not None else None
            ),
            phis_end_cr=(
                _tree_cumsum_last(dphis_end_cr)
                if dphis_end_cr is not None else None
            ),
            v_end_cr=v_end_cr,
            x_foundation=x_foundation,
            thetas_orientation_cc=thetas_orientation_cc,
            thetas_orientation_cf=thetas_orientation_cf,
            thetas_orientation_cr=thetas_orientation_cr,
            csr_curve_dofs=csr_curve_dofs,
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
        renamed = [_PHI_TO_DPHI.get(k, k) for k in fixed_dof_names]
        return encoded, renamed
