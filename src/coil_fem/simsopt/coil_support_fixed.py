"""Fixed-clamp simsopt coil support models.

:class:`CoilSupportFixed`, :class:`CoilSupportTopBottom`, and
:class:`CoilSupportFixedSorted` — grounded Winkler clamps without beam coupling.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..utils import clamp_sigmoid
from ..coupling import Support
from .coil_support import (
    CoilSupport,
    _SortedDphisMixin,
    _broadcast_phis,
    _cumsum_last,
    _diff_last,
    _generate_k_clamp,
)


class CoilSupportFixed(CoilSupport):
    """Support modelled as ``n_clamp`` clamps at optimisable arclengths.

    Each clamp is a sphere of radius ``r_clamp`` centred on the coil at
    curve parameter ``phi``; the per-node weight is the (smooth) union of all
    clamp indicator spheres.  Only the clamp locations ``phis`` are DOFs;
    ``r_clamp``, ``eps_sigmoid`` and ``n_clamp`` are fixed.

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
        k_clamp : float
            Grounded Winkler spring modulus [N/m³].
        r_clamp : float
            Sphere radius [m] of each clamp (region of non-zero spring weight).
        n_clamp : int
            Number of clamps per coil.
        eps_sigmoid : float
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
        k_clamp = _generate_k_clamp(base_coils, fixed_clamp_options)
        try:
            r_clamp = fixed_clamp_options['r_clamp']
            n_clamp = fixed_clamp_options['n_clamp']
        except KeyError:
            raise KeyError(
                "fixed_clamp_options must contain 'r_clamp' and 'n_clamp'."
            )
        eps_sigmoid = fixed_clamp_options.get('eps_sigmoid', 0.1)

        phis_arr = _broadcast_phis(phis, n_coils, n_clamp)

        self._r_clamp = float(r_clamp)
        self._sig_eps  = float(eps_sigmoid)

        # ── Stored for GSONable serialization ────────────────────────────────────
        # GSONable.as_dict requires every __init__ parameter to be present as
        # self.<param> or self._<param>.
        self._fixed_clamp_options = {
            'k_clamp': k_clamp, 'r_clamp': r_clamp,
            'n_clamp': n_clamp, 'eps_sigmoid': eps_sigmoid,
        }
        # Serialization seeds only — actual clamp positions are restored from the
        # Optimizable dofs object by simsopt on load.
        self._phis  = phis
        self._names = names
        # ─────────────────────────────────────────────────────────────────────────

        support_dofs_jax = self._angle_support_dofs(phis_arr)
        lb, ub = self._make_bounds(
            support_dofs_jax,
            unit_interval_keys=tuple(support_dofs_jax),
        )
        super().__init__(
            base_coils, nfp, stellsym,
            support=Support(k_clamp=float(k_clamp), fixed_clamp_fns=self._clamp_fn),
            support_dofs_jax=support_dofs_jax,
            constants={
                'r_clamp': float(r_clamp),
                'eps_sigmoid': float(eps_sigmoid),
            },
            names=names,
            lower_bounds=lb,
            upper_bounds=ub,
            dofs=dofs,
        )

    def _angle_support_dofs(self, phis_arr):
        """Build the angle DOF pytree from absolute clamp locations."""
        return {'phis': phis_arr}

    def _clamp_fn(self, surface_pts, curve_jax, dofs_slice):
        """Winkler weight for one coil; dispatched by Support.compute_weights."""
        if not dofs_slice:
            return jnp.zeros(surface_pts.shape[0])
        phis_i = dofs_slice['phis']
        gamma_support = curve_jax.gamma_eval(phis_i)
        d_sq = jnp.sum(
            (surface_pts[:, None, :] - gamma_support[None, :, :]) ** 2,
            axis=-1,
        )
        w = clamp_sigmoid(d_sq, self._r_clamp, self._sig_eps)
        return jnp.sum(w, axis=-1)


class CoilSupportFixedSorted(_SortedDphisMixin, CoilSupportFixed):
    """:class:`CoilSupportFixed` with incremental ``dphis`` DOFs.

    Simsopt stores ``dphis`` with ``phis = cumsum(dphis)`` along the clamp
    axis.  :attr:`support_dofs` exposes absolute ``phis`` for the FEM;
    :meth:`flatten_grad` applies the cumsum VJP so ``dJ`` is w.r.t. ``dphis``.

    Parameters
    ----------
    base_coils, nfp, stellsym, fixed_clamp_options
        Same as :class:`CoilSupportFixed`.
    dphis : array-like or None
        Initial increments, shape ``(n_coils, n_clamp)`` or ``(n_clamp,)``.
        ``None`` uses the same default clamp layout as
        :class:`CoilSupportFixed` (encoded via ``diff``).
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
        dphis=None,
        names=None,
        dofs=None,
    ):
        self._dphis = dphis
        if dphis is not None:
            n_coils = len(base_coils)
            n_clamp = fixed_clamp_options['n_clamp']
            phis = _cumsum_last(_broadcast_phis(dphis, n_coils, n_clamp))
        else:
            phis = None
        super().__init__(
            base_coils, nfp, stellsym, fixed_clamp_options,
            phis=phis, names=names, dofs=dofs,
        )

    def _angle_support_dofs(self, phis_arr):
        return {'dphis': _diff_last(phis_arr)}


class CoilSupportTopBottom(CoilSupport):
    """Static soft-sphere support at the top and bottom of the coil centreline.

    Has no optimisable DOFs (``support_dofs_jax={}``); ``r_clamp`` and
    ``eps_sigmoid`` are fixed constants.

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
        k_clamp : float
            Grounded Winkler spring modulus [N/m³].
        r_clamp : float
            Sphere radius [m] of each clamp (region of non-zero spring weight).
        eps_sigmoid : float
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
        k_clamp = _generate_k_clamp(base_coils, fixed_clamp_options)
        try:
            r_clamp = fixed_clamp_options['r_clamp']
        except KeyError:
            raise KeyError(
                "fixed_clamp_options must contain 'r_clamp'."
            )
        eps_sigmoid = fixed_clamp_options.get('eps_sigmoid', 0.1)

        self._r_clamp = float(r_clamp)
        self._sig_eps  = float(eps_sigmoid)

        # ── Stored for GSONable serialization ────────────────────────────────────
        # GSONable.as_dict requires every __init__ parameter to be present as
        # self.<param> or self._<param>.
        self._fixed_clamp_options = {
            'k_clamp': k_clamp, 'r_clamp': r_clamp, 'eps_sigmoid': eps_sigmoid,
        }
        # ─────────────────────────────────────────────────────────────────────────

        super().__init__(
            base_coils, nfp, stellsym,
            support=Support(k_clamp=float(k_clamp), fixed_clamp_fns=self._clamp_fn),
            support_dofs_jax={},
            constants={
                'r_clamp': float(r_clamp),
                'eps_sigmoid': float(eps_sigmoid),
            },
            dofs=dofs,
        )

    def _clamp_fn(self, surface_pts, curve_jax, dofs_slice):
        """Winkler weight at the coil's topmost and bottommost points."""
        gamma = curve_jax.gamma()
        top = gamma[jnp.argmax(gamma[:, 2])]
        bottom = gamma[jnp.argmin(gamma[:, 2])]
        w_top = clamp_sigmoid(
            jnp.sum((surface_pts - top) ** 2, axis=-1),
            self._r_clamp, self._sig_eps,
        )
        w_bottom = clamp_sigmoid(
            jnp.sum((surface_pts - bottom) ** 2, axis=-1),
            self._r_clamp, self._sig_eps,
        )
        return w_top + w_bottom


