"""Incremental ``dphis*`` ↔ ``phis*`` codecs for Sorted coil-support classes.

Provides the name maps, last-axis (beam) and coil-axis (CR-end) diff/cumsum
primitives, pytree encode / decode / VJP, first-increment fold and box
bounds, the stellsym wrap-end ``dphis_end_cc`` codec, and
:class:`_SortedDphisMixin`.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map


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
# Increments run along the last (beam) axis.
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


def _axis_slice(shape, axis, rest):
    """Index tuple selecting the first (or rest) entries along ``axis``."""
    axis = axis if axis >= 0 else len(shape) + axis
    sl = [slice(None)] * len(shape)
    sl[axis] = slice(1, None) if rest else 0
    return tuple(sl)


def _axis_mask(tree, keys, axis=-1, rest=False):
    """Boolean pytree: True on the first (or rest) slice along ``axis``.

    Applied only to leaves of ``keys``.  ``axis`` is the increment axis:
    last (beam) for half-turn keys, ``0`` (coil) for ``dphis_end_cr``.
    """
    keys = set(keys)

    def _leaf(leaf, key):
        mask = np.zeros(np.shape(leaf), dtype=bool)
        if key not in keys or mask.ndim < 1:
            return mask
        n = mask.shape[axis]
        if rest:
            if n > 1:
                mask[_axis_slice(mask.shape, axis, True)] = True
        elif n > 0:
            mask[_axis_slice(mask.shape, axis, False)] = True
        return mask

    return {
        k: tree_map(lambda leaf, kk=k: _leaf(leaf, kk), v)
        for k, v in tree.items()
    }


def _fold_first_along_axis(x, lo, hi, axis):
    """Fold only the first slice of ``x`` along ``axis`` into ``[lo, hi]``."""
    x = jnp.asarray(x, dtype=float)
    if x.ndim == 0 or x.shape[axis] == 0:
        return x
    sl = _axis_slice(x.shape, axis, rest=False)
    return x.at[sl].set(_fold_into_interval(x[sl], lo, hi))


def _fold_first_dphis(tree, nfp, stellsym):
    """Fold first Sorted increments into their box so default ``x0`` is feasible.

    Half-turn keys (CC start/end, CF start, CR start) fold beam index 0
    into ``[-0.5, 0.5]``.  ``dphis_end_cr`` folds coil index 0 (row 0 of
    the ``(n_coil, n_beam_cr)`` array) into ``[-0.5 s, 0.5 s]`` with
    ``s = :func:`_sector_width``.  Later increments and other keys are
    left alone.  Clamp ``dphis`` is not folded.
    """
    s = _sector_width(nfp, stellsym)
    out = {}
    for k, v in tree.items():
        if k in _FIRST_DPHI_HALF_TURN_KEYS:
            out[k] = tree_map(
                lambda leaf: _fold_first_along_axis(leaf, -0.5, 0.5, -1), v,
            )
        elif k == 'dphis_end_cr':
            out[k] = _fold_first_along_axis(v, -0.5 * s, 0.5 * s, 0)
        else:
            out[k] = v
    return out


def _apply_sorted_dphi_bounds(lb, ub, tree, nfp, stellsym):
    """Overlay Sorted first/rest ``dphis*`` box bounds on flattened ``lb``/``ub``.

    No-op when ``tree`` has no ``dphis*`` keys (non-Sorted ``phis*`` classes).

    * first of :data:`_FIRST_DPHI_HALF_TURN_KEYS` (beam axis): ``[-0.5, 0.5]``
    * coil 0 of ``dphis_end_cr`` (row 0): ``[-0.5 s, 0.5 s]``
    * later coils of ``dphis_end_cr`` (rows ``1:``): upper bound ``s``
      (lower stays 0)
    """
    if not any(k.startswith('dphis') for k in tree):
        return lb, ub
    s = _sector_width(nfp, stellsym)
    first_half, _ = ravel_pytree(
        _axis_mask(tree, _FIRST_DPHI_HALF_TURN_KEYS, axis=-1),
    )
    first_cr, _ = ravel_pytree(
        _axis_mask(tree, ('dphis_end_cr',), axis=0),
    )
    rest_cr, _ = ravel_pytree(
        _axis_mask(tree, ('dphis_end_cr',), axis=0, rest=True),
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


# ============================================================================
# CSR-end codec: increments along the coil axis (axis 0)
# ============================================================================

def _decode_end_cr(d):
    """Decode ``dphis_end_cr`` → ``phis_end_cr`` (cumsum over coils)."""
    return jnp.cumsum(jnp.asarray(d, dtype=float), axis=0)


def _encode_end_cr(phi):
    """Encode ``phis_end_cr`` → ``dphis_end_cr`` (diff over coils)."""
    phi = jnp.asarray(phi, dtype=float)
    return jnp.diff(phi, axis=0, prepend=jnp.zeros_like(phi[:1]))


def _vjp_end_cr(g):
    """VJP of :func:`_decode_end_cr`: reverse-cumsum over coils."""
    g = jnp.asarray(g, dtype=float)
    return jnp.cumsum(g[::-1], axis=0)[::-1]


def _encode_dphis(phi_dofs: dict) -> dict:
    """Functional ``phis*`` pytree → stored ``dphis*`` pytree (diff + rename).

    ``phis_end_cr`` diffs along the coil axis; every other ``phis*`` key
    diffs along the last (beam) axis.
    """
    out = {}
    for k, v in phi_dofs.items():
        if k == 'phis_end_cr':
            out['dphis_end_cr'] = _encode_end_cr(v)
        elif k in _PHI_TO_DPHI:
            out[_PHI_TO_DPHI[k]] = tree_map(_diff_last, v)
        else:
            out[k] = v
    return out


def _decode_dphis(raw_dofs: dict) -> dict:
    """Stored ``dphis*`` pytree → functional ``phis*`` pytree (cumsum + rename).

    ``dphis_end_cr`` cumsums along the coil axis; every other ``dphis*``
    key cumsums along the last (beam) axis.
    """
    out = {}
    for k, v in raw_dofs.items():
        if k == 'dphis_end_cr':
            out['phis_end_cr'] = _decode_end_cr(v)
        elif k in _DPHI_TO_PHI:
            out[_DPHI_TO_PHI[k]] = tree_map(_cumsum_last, v)
        else:
            out[k] = v
    return out


def _vjp_dphis(grad_phi_dofs: dict) -> dict:
    """Grad w.r.t. ``phis*`` → grad w.r.t. ``dphis*`` (cumsum VJP + rename).

    ``phis_end_cr`` uses the coil-axis VJP; every other ``phis*`` key uses
    the last-axis VJP.
    """
    out = {}
    for k, v in grad_phi_dofs.items():
        if k == 'phis_end_cr':
            out['dphis_end_cr'] = _vjp_end_cr(v)
        elif k in _PHI_TO_DPHI:
            out[_PHI_TO_DPHI[k]] = tree_map(_cumsum_last_vjp, v)
        else:
            out[k] = v
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


class _SortedDphisMixin:
    """Decode stored ``dphis*`` for the FEM; pull grads back in ``flatten_grad``.

    When ``dphis_end_cc`` is present, the two stellsym wrap groups use
    ``phis_end = 1 - cumsum(dphis_end)`` (see
    :func:`~coil_fem.simsopt.sorted_dphis._decode_end_cc`).  Subclasses
    that use that key must set ``_sorted_stellsym`` before the first
    ``support_dofs`` / ``flatten_grad`` call.

    ``dphis_end_cr`` is decoded by cumsum over the coil axis (see
    :func:`~coil_fem.simsopt.sorted_dphis._decode_end_cr`).
    """

    @property
    def support_dofs(self) -> dict:
        raw = self._unravel(jnp.asarray(self.local_full_x))
        out = _decode_dphis(raw)
        if 'dphis_end_cc' in raw:
            out['phis_end_cc'] = _decode_end_cc(
                raw['dphis_end_cc'], self._sorted_stellsym,
            )
        return out

    def flatten_grad(self, grad_dofs: dict) -> np.ndarray:
        g = _vjp_dphis(grad_dofs)
        if 'phis_end_cc' in grad_dofs:
            g['dphis_end_cc'] = _vjp_end_cc(
                grad_dofs['phis_end_cc'], self._sorted_stellsym,
            )
        return np.asarray(ravel_pytree(g)[0], dtype=float)
