# RMF frame smoothness — finding, fix, and API break

Status: fixed 2026-08-04.  Supersedes the grounded-`w_a` investigation, whose
hypotheses were all wrong; its notes (`WINKLER_WA_VJP.md`, `autodiff_fixes.md`)
and the `COIL_FEM_VJP_ABLATION` instrumentation they described have been removed.

## The finding

The W7-X SupportBeams monolithic Taylor test was short by ~1.2%
(`dJh/FD = 0.98834`) with an **ε-independent** bias across 1e-5 → 1e-7.  That
signature is a kink, not a missing term.

`SupportBeams._frame_at_phi` interpolated the coil cross-section frame from the
curve's native quadpoint grid with **periodic linear interpolation** (`jnp.floor`)
— C0 only, with a kink at every quadpoint.  Everything else in the same geometry
pipeline used the exact Fourier `gamma_eval`.  The frame fed
`_xi_surface_exit` → `xi_start`/`xi_end` → `L_eff` → beam stiffness.

Two facts made it bite hard:

1. All 40 attachment angles sit **exactly on quadpoint knots**.  `get_data('w7x',
   points_per_period=8)` gives N=64, and `_uniform_list`
   (`simsopt/optimizables.py`) puts the defaults at odd multiples of 1/8 —
   0.125·64 = 8, exactly integer.  At a knot, AD returns the right-hand chord
   slope while a centered FD returns the mean of both sides; the difference is
   O(1/N) and does not shrink with ε.
2. Measured error on `dL_eff/dphi`: median ~3%, worst 203%.

Diagnostics: `fem-data/beams-qss/geom_smoothness.py` (D1 knot alignment,
D2 one-sided/centered FD triad, D3 kink scan).

### Why the earlier investigation missed it

Every "block is healthy" result was below the resolution needed.  The gap was
3.6e-7 *relative* on blocks of size 1e21, while the probes used a single ε=1e-7
FD and a 5e-3 pass threshold.  The decisive datum was already in the logs and was
read backwards: `R_full (λ cotangent)` gave `an/FD = 0.98834` with `u*` **and**
`λ` frozen — i.e. a pure-function AD-vs-FD discrepancy with no solve, no adjoint
and no `custom_vjp`, which excludes every composition-flavoured hypothesis.

## The fix

`geo/framed_curve_jax.py`:

- The RMF is stored as a **scalar twist angle** relative to the closed-form
  centroid frame (`_rmf_twist_pure`), built once in `__init__` and carried as a
  pytree child.  `t` comes from `gamma_eval` and is exact; `p`/`q` are exactly
  orthonormal by construction because only a scalar is interpolated.
- Interpolation is band-limited trigonometric (`_trig_interp`, factored out of
  `alpha_eval`).  The RMF's closure correction leaves an O(1/N) residual, so the
  angle is split into an analytic linear winding ramp plus an exactly periodic
  remainder (`_unwrap_periodic`) — no Gibbs ringing at the `phi = 0` seam.
- `tree_unflatten` passes the stored twist straight through, so pytree
  round-trips never re-run the `lax.scan`.  `with_dofs` deliberately omits it, so
  new DOFs rebuild exactly once — the single construction point per DOF update.

`coupling/beam_network.py`: `_frame_at_phi` deleted; `_surface_exit_params` calls
`rotated_frame_eval`.  Frame logic no longer lives in the support module.

## API break (no deprecation shims)

`FramedCurveRMFJAX.rotated_frame_eval` no longer rebuilds the RMF on the supplied
grid.  It interpolates the frame built at construction, so it is smooth and
**independent of the ordering and density of `phi`** — scattered `phi` is now
fully supported.  `FramedCurveCentroidJAX.rotated_frame_eval` is unchanged
(already closed-form and exact).

`FramedCurveJAX.__init__` takes a keyword-only `twist=`; `tree_flatten` now
returns three children.

## Verification

`fem-data/beams-qss/testing.py` (`sbatch jobscript_testing.sh`), job 15275557:

| Check | Result |
|---|---|
| Twist reconstructs the original scan frame | 9.2e-16 |
| Off-grid orthonormality | 2.6e-16 |
| Order/density independence of `phi` | exact |
| Frame built once (pytree round-trip, `jit`) | 0 rebuilds; 1 per `with_dofs` |
| **W7-X geometry chain, all 40 dofs** | **0 kinked; worst `\|AD-bwd\|` 7.2e-06** (was 4.07, flat) |

## Accepted behaviour change: mesh geometry

`meshing._rect_sweep_points` used to get an RMF **re-swept on the refined
`K = 2M` grid** for TET10; it now interpolates the native-`M` frame.  Measured on
the W7-X coilset (job 15275557):

| Coil | max frame angle diff | max node shift |
|---|---|---|
| 0 | 1.416e-02 rad | 1.416e-03 m |
| 1 | 1.405e-02 rad | 1.405e-03 m |
| 2 | **1.727e-02 rad** | **1.727e-03 m** |
| 3 | 1.076e-02 rad | 1.076e-03 m |
| 4 | 3.392e-03 rad | 3.392e-04 m |

Worst case ~1.7 mm against a 0.2 m cross-section (1.7% of the half-width), on the
scale of the RMF's own closure accuracy `1/M = 1.56e-02`.  Meshes and any
baseline tied to them shift; re-base deliberately.

Note the compensating gain: previously the **mesh** used the `K = 2M` re-sweep
while the **beam attachments** used a linear interpolation of the `M`-grid frame —
two different frames for the same coil.  They now agree.

To shrink the difference, raise `points_per_period` (M=64 → 128): it scales the
frame difference down ~2× *and* improves the RMF's own closure residual.

## One frame definition per class

Follow-up cleanup: `rotated_frame_eval` is now the single frame definition, and
`rotated_frame()` and `_rotated_frame_and_dash()` both derive from it —
the latter as a plain `jax.jvp` in `phi`, since the evaluator is smooth.  Deleted:
`_rotated_centroid_frame_pure`, `_rotated_rmf_frame_pure`, `_frame_pure_fn`,
`_rotated_frame_fn`, `_frame_jitted`, `_frame_and_dash_jitted`, `alphadash()`.
`_centroid_reference` replaces three near-identical copies of the centroid-frame
construction.  The `lax.scan` now runs only in `FramedCurveRMFJAX.__init__`.

### Measured consequence: RMF torsion

`frame_curvatures()` for RMF frames now returns the torsion of the interpolated
frame rather than of the discrete propagation (W7-X coils, vs pre-cleanup):

| Quantity | Centroid | RMF |
|---|---|---|
| κ1, κ2 | unchanged (≤2.3e-15 rel) | unchanged (≤1.8e-15 rel) |
| κ3 (torsion) | unchanged (1.5e-15 rel) | **changed** |

Absolute κ3, worst per coil: `2.9e-03 → 2.2e-01`, `3.3e-02 → 2.3e-01`,
`8.0e-02 → 2.5e-01`, `2.2e-01 → 2.1e-01`, `3.3e-01 → 1.6e-01` — so it grows on
some coils and shrinks on others, against κ1 ≈ 2.0 throughout.

**No consumer in the library.** `magnetic.py:286` takes only κ1 and κ2
(`kappa1_cl, kappa2_cl, _ = frame_curvatures()`), both unchanged to 1e-15, so
`B_self_quadrature` is unaffected.  Nothing else in `src/` reads `frame_torsion`.

The new value is the more honest one: it describes the frame the mesh and the
beam attachments actually use.  Before, the mesh came from a `K = 2M` re-sweep
while κ3 described the `M`-grid scan — neither described the other.

## Known-adjacent, not fixed

`_rmf_normals_pure_jax` distributes the periodic closure correction with
`jnp.linspace(0., angle_corr, N)`, which applies the *full* correction at index
`N-1` rather than at the wrap point.  With `phi_k = k/N`, exact periodicity wants
`jnp.arange(N) * angle_corr / N`.  This is the source of the O(1/N) residual gap
in the class docstring.  Left alone deliberately: fixing it changes the frame
itself and should not ride along with an AD-correctness refactor.  The ramp split
in `_unwrap_periodic` makes the refactor correct either way.
