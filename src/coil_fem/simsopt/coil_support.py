"""Simsopt base container for coil support DOFs.

:class:`CoilSupport` holds the base coils, ``nfp``, ``stellsym``, and a
pure-functional :class:`~coil_fem.coupling.Support` instance.  Optimisable
parameters live in the simsopt DOF store; :attr:`CoilSupport.support_dofs`
reconstructs them from ``local_full_x``.

Also provides shared helpers used by Fixed/Beams subclasses (angle broadcast,
``k_clamp`` defaults, and incremental ``dphis*`` encode/decode).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import (
    tree_map,
    tree_flatten_with_path,
    DictKey,
    SequenceKey,
)
from ..utils import estimate_k
from ..coupling import Support



# ============================================================================
# Sorted (incremental) angle DOFs: dphis* ↔ phis*
# ============================================================================

_DPHI_TO_PHI = {
    'dphis': 'phis',
    'dphis_start_cc': 'phis_start_cc',
    'dphis_end_cc': 'phis_end_cc',
    'dphis_start_cf': 'phis_start_cf',
    'dphis_start_cr': 'phis_start_cr',
    'dphis_end_cr': 'phis_end_cr',
}
_PHI_TO_DPHI = {v: k for k, v in _DPHI_TO_PHI.items()}
_ANGLE_UNIT_KEYS = frozenset(_DPHI_TO_PHI) | frozenset(_PHI_TO_DPHI)

# First-increment box is [-0.5, 0.5] for these Sorted keys (period 1).
_FIRST_DPHI_HALF_TURN_KEYS = frozenset({
    'dphis_start_cc',
    'dphis_end_cc',
    'dphis_start_cf',
    'dphis_start_cr',
})


def _sector_width(nfp, stellsym):
    """CSR-end increment period: ``1/nfp``, or ``1/(2 nfp)`` under stellsym."""
    return 1.0 / float(nfp) / (2.0 if stellsym else 1.0)


def _fold_into_interval(x, lo, hi):
    """Periodic reduction of ``x`` into ``[lo, hi)`` (``hi`` maps to ``lo``)."""
    width = hi - lo
    return jnp.mod(x - lo, width) + lo


def _fold_first_last_axis(x, lo, hi):
    """Fold only ``x[..., 0]`` into ``[lo, hi]``; later increments unchanged."""
    x = jnp.asarray(x, dtype=float)
    if x.ndim == 0 or x.shape[-1] == 0:
        return x
    return x.at[..., 0].set(_fold_into_interval(x[..., 0], lo, hi))


def _fold_first_dphis(tree, nfp, stellsym):
    """Fold first Sorted increments into their box so default ``x0`` is feasible.

    Half-turn keys (CC start/end, CF start, CR start) fold into ``[-0.5, 0.5]``.
    ``dphis_end_cr`` folds into ``[-0.5 s, 0.5 s]`` with
    ``s = :func:`_sector_width``.  Later increments and other keys are left
    alone.  Clamp ``dphis`` is not folded.
    """
    s = _sector_width(nfp, stellsym)
    out = {}
    for k, v in tree.items():
        if k in _FIRST_DPHI_HALF_TURN_KEYS:
            out[k] = tree_map(
                lambda leaf: _fold_first_last_axis(leaf, -0.5, 0.5), v,
            )
        elif k == 'dphis_end_cr':
            out[k] = tree_map(
                lambda leaf: _fold_first_last_axis(leaf, -0.5 * s, 0.5 * s), v,
            )
        else:
            out[k] = v
    return out


def _last_axis_first_mask(tree, keys):
    """Boolean pytree: True on ``leaf[..., 0]`` for ``keys``."""
    keys = set(keys)

    def _leaf(leaf, key):
        mask = np.zeros(np.shape(leaf), dtype=bool)
        if key in keys and mask.ndim >= 1 and mask.shape[-1] > 0:
            mask[..., 0] = True
        return mask

    return {
        k: tree_map(lambda leaf, kk=k: _leaf(leaf, kk), v)
        for k, v in tree.items()
    }


def _last_axis_rest_mask(tree, keys):
    """Boolean pytree: True on ``leaf[..., 1:]`` for ``keys``."""
    keys = set(keys)

    def _leaf(leaf, key):
        mask = np.zeros(np.shape(leaf), dtype=bool)
        if key in keys and mask.ndim >= 1 and mask.shape[-1] > 1:
            mask[..., 1:] = True
        return mask

    return {
        k: tree_map(lambda leaf, kk=k: _leaf(leaf, kk), v)
        for k, v in tree.items()
    }


def _apply_sorted_dphi_bounds(lb, ub, tree, nfp, stellsym):
    """Overlay Sorted first/rest ``dphis*`` box bounds on flattened ``lb``/``ub``.

    No-op when ``tree`` has no ``dphis*`` keys (non-Sorted ``phis*`` classes).

    * first of :data:`_FIRST_DPHI_HALF_TURN_KEYS`: ``[-0.5, 0.5]``
    * first of ``dphis_end_cr``: ``[-0.5 s, 0.5 s]``
    * remaining ``dphis_end_cr``: upper bound ``s`` (lower stays 0)
    """
    if not any(k.startswith('dphis') for k in tree):
        return lb, ub
    s = _sector_width(nfp, stellsym)
    first_half, _ = ravel_pytree(
        _last_axis_first_mask(tree, _FIRST_DPHI_HALF_TURN_KEYS),
    )
    first_cr, _ = ravel_pytree(
        _last_axis_first_mask(tree, ('dphis_end_cr',)),
    )
    rest_cr, _ = ravel_pytree(
        _last_axis_rest_mask(tree, ('dphis_end_cr',)),
    )
    lb = np.asarray(lb, dtype=float).copy()
    ub = np.asarray(ub, dtype=float).copy()
    lb = np.where(first_half, -0.5, lb)
    ub = np.where(first_half, 0.5, ub)
    lb = np.where(first_cr, -0.5 * s, lb)
    ub = np.where(first_cr, 0.5 * s, ub)
    ub = np.where(rest_cr, s, ub)
    return lb, ub


def _diff_last(x):
    """Encode absolute angles as increments along the last axis."""
    return jnp.diff(x, axis=-1, prepend=jnp.zeros_like(x[..., :1]))


def _cumsum_last(x):
    """Decode increments to absolute angles along the last axis."""
    return jnp.cumsum(x, axis=-1)


def _cumsum_last_vjp(g):
    """VJP of :func:`_cumsum_last`: ``g_d[..., j] = sum_{k>=j} g_phi[..., k]``."""
    return jnp.cumsum(g[..., ::-1], axis=-1)[..., ::-1]


def _tree_diff_last(tree):
    return tree_map(_diff_last, tree)


def _tree_cumsum_last(tree):
    return tree_map(_cumsum_last, tree)


def _tree_cumsum_last_vjp(g_tree):
    return tree_map(_cumsum_last_vjp, g_tree)


def _encode_dphis(phi_dofs: dict) -> dict:
    """Functional ``phis*`` pytree → stored ``dphis*`` pytree (diff + rename)."""
    out = {}
    for k, v in phi_dofs.items():
        if k in _PHI_TO_DPHI:
            out[_PHI_TO_DPHI[k]] = _tree_diff_last(v)
        else:
            out[k] = v
    return out


def _decode_dphis(raw_dofs: dict) -> dict:
    """Stored ``dphis*`` pytree → functional ``phis*`` pytree (cumsum + rename)."""
    out = {}
    for k, v in raw_dofs.items():
        if k in _DPHI_TO_PHI:
            out[_DPHI_TO_PHI[k]] = _tree_cumsum_last(v)
        else:
            out[k] = v
    return out


def _vjp_dphis(grad_phi_dofs: dict) -> dict:
    """Grad w.r.t. ``phis*`` → grad w.r.t. ``dphis*`` (cumsum VJP + rename)."""
    out = {}
    for k, v in grad_phi_dofs.items():
        if k in _PHI_TO_DPHI:
            out[_PHI_TO_DPHI[k]] = _tree_cumsum_last_vjp(v)
        else:
            out[k] = v
    return out


class _SortedDphisMixin:
    """Decode stored ``dphis*`` for the FEM; pull grads back in ``flatten_grad``.

    When ``dphis_end_cc`` is present, the two stellsym wrap groups use
    ``phis_end = 1 - cumsum(dphis_end)`` (see
    :func:`~coil_fem.simsopt.coil_support_beams._decode_end_cc`).  Subclasses
    that use that key must set ``_sorted_stellsym`` before the first
    ``support_dofs`` / ``flatten_grad`` call.
    """

    @property
    def support_dofs(self) -> dict:
        raw = self._unravel(jnp.asarray(self.local_full_x))
        out = _decode_dphis(raw)
        if 'dphis_end_cc' in raw:
            # Lazy import avoids a circular dependency with coil_support_beams.
            from .coil_support_beams import _decode_end_cc
            out['phis_end_cc'] = _decode_end_cc(
                raw['dphis_end_cc'], self._sorted_stellsym,
            )
        return out

    def flatten_grad(self, grad_dofs: dict) -> np.ndarray:
        g = _vjp_dphis(grad_dofs)
        if 'phis_end_cc' in grad_dofs:
            from .coil_support_beams import _vjp_end_cc
            g['dphis_end_cc'] = _vjp_end_cc(
                grad_dofs['phis_end_cc'], self._sorted_stellsym,
            )
        return np.asarray(ravel_pytree(g)[0], dtype=float)


# ============================================================================
def _generate_k_clamp(base_coils, fixed_clamp_options):
    """ Defaults for the fixed clamp's Robin/Winkler spring coefficients.
    
    A fixed clamp's Robin/Winkler spring coefficients can be auto-generated
    based on the beam's stiffness. This method generates them from the coil's 
    stiffness. 
    """
    if 'k_clamp' in fixed_clamp_options.keys():
        return fixed_clamp_options['k_clamp']
    else:
        # The beams can bend and therefore can numerically
        # tolerate a much higher regularization factor than
        # the fixed clamps. A higher factor by at least 2^4 is 
        # needed because beams can be 2x more narrow than coils.
        # TODO: it may be possible to change this factor 
        # dynamically in an optimization based on the 
        # support beam thickness.
        eps_clamp = fixed_clamp_options.get('eps_clamp', 1e-5)
        try:
            E_coil = fixed_clamp_options['E_coil']
        except:
            raise KeyError(
                "E_coil not detected in fixed_clamp_options. "
                "When k_clamp is not provided, the Young's modulus "
                "of the coils, E_coil, must be provided so that a the value "
                "of k_clamp can be auto-generated based on the coil's stiffness."
            )
        mean_arclengths = np.mean(
            # This calculates the total length of each coil
            [jnp.mean(c.curve.incremental_arclength()) for c in base_coils]
        )
        L_coil = mean_arclengths / np.pi / 2
        k_clamp = estimate_k(L=L_coil, E=E_coil, eps=eps_clamp)
        print(
            "k_clamp is not provided. Based on the coil's stiffness, "
            f"the auto-generated value is      {k_clamp:.4e} N/m3."
        )
        return k_clamp


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
    lower_bounds : array-like or None
        Lower bounds aligned with the flattened DOF vector (default ``-inf``).
    upper_bounds : array-like or None
        Upper bounds aligned with the flattened DOF vector (default ``+inf``).
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
        lower_bounds=None,
        upper_bounds=None,
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
            if names is None:
                names = self._make_names(support_dofs_jax)
            Optimizable.__init__(
                self,
                x0=np.array(flat, dtype=float),
                names=names,
                fixed=fixed,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
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

    def _make_names(self, support_dofs_jax: dict) -> list[str]:
        """Build simsopt DOF names from a ``support_dofs`` pytree path.

        Names follow the flatten order of :func:`jax.flatten_util.ravel_pytree`.
        List/array indices are written as ``key(i,j,...)``.

        Parameters
        ----------
        support_dofs_jax : dict
            Optimisable support DOF pytree (same structure as
            :attr:`support_dofs`).

        Returns
        -------
        list of str
            One name per scalar in the flattened DOF vector.
        """
        names: list[str] = []
        paths_and_leaves, _ = tree_flatten_with_path(support_dofs_jax)
        for path, leaf in paths_and_leaves:
            key = None
            prefix: list[int] = []
            for part in path:
                if isinstance(part, DictKey):
                    key = part.key
                elif isinstance(part, SequenceKey):
                    prefix.append(part.idx)
            assert key is not None
            arr = np.asarray(leaf)
            for idx in np.ndindex(arr.shape):
                full = (*prefix, *idx)
                names.append(f"{key}({','.join(map(str, full))})")
        return names

    def _make_bounds(
        self,
        support_dofs_jax: dict,
        unit_interval_keys=(),
        nonnegative_keys=(),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build full-length lower/upper bounds for the Optimizable based on keys in support_dofs.

        Bounds are raveled in the same order as
        :func:`jax.flatten_util.ravel_pytree` / :meth:`_make_names`.

        Parameters
        ----------
        support_dofs_jax : dict
            Optimisable support DOF pytree.
        unit_interval_keys : iterable of str
            Keys bounded to ``[0, 1]``.
        nonnegative_keys : iterable of str
            Keys bounded to ``[0, +inf)``.

        Returns
        -------
        lower_bounds, upper_bounds : ndarray
            Full-length bound arrays aligned with the flattened DOF vector.
        """
        unit = set(unit_interval_keys)
        nonneg = set(nonnegative_keys)

        def _lb_leaf(leaf, key):
            shape = np.shape(leaf)
            if key in unit or key in nonneg:
                return np.zeros(shape, dtype=float)
            return np.full(shape, -np.inf, dtype=float)

        def _ub_leaf(leaf, key):
            shape = np.shape(leaf)
            if key in unit:
                return np.ones(shape, dtype=float)
            return np.full(shape, np.inf, dtype=float)

        lb_tree = {
            k: tree_map(lambda leaf, kk=k: _lb_leaf(leaf, kk), v)
            for k, v in support_dofs_jax.items()
        }
        ub_tree = {
            k: tree_map(lambda leaf, kk=k: _ub_leaf(leaf, kk), v)
            for k, v in support_dofs_jax.items()
        }
        lb, _ = ravel_pytree(lb_tree)
        ub, _ = ravel_pytree(ub_tree)
        return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def _broadcast_phis(phis, n_coils, n_clamp):
    if phis is None:
        row = jnp.linspace(0.0, 1.0, n_clamp, endpoint=False)
        phis_arr = jnp.broadcast_to(row, (n_coils, n_clamp))
    else:
        phis_arr = jnp.asarray(phis, dtype=float)
        if phis_arr.ndim == 1:
            phis_arr = jnp.broadcast_to(phis_arr, (n_coils, n_clamp))
    return phis_arr


