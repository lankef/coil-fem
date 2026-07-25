# coil-fem code review — July 2026

Review of every module under `src/coil_fem/` against the four rules in the
review brief (YAGNI, duplication, replaceable-by-dependency, JIT) plus the six
architectural philosophy points.

Reviewed at `beam-network` @ `90f5bcb` with the working-tree changes applied
(`coil_fem.py`, `coupling/{beam_network,drivers,supports}.py`,
`problems/linear_elasticity.py`, `tests/test_beam_networks.py`,
`tests/test_surface_quadrature.py`).

Everything marked **[measured]** was verified by running code in the `rod`
environment; see [Appendix A](#appendix-a--measurement-setup) for the setup.
Everything else is from reading the source and cross-referencing call sites
across `src/`, `tests/`, `examples/`, and `docs/`.

---

## Executive summary

Three findings are worth acting on before anything cosmetic:

1. **`jax.grad` through the staggered driver does not work.** It raises
   `ConcretizationTypeError`, and the `custom_vjp` that the docstring
   advertises as providing implicit-function-theorem gradients is bypassed
   even in principle. **[measured]**
2. **The staggered driver is dominated by un-jitted JAX dispatch, not by the
   FEM solves it exists to coordinate.** A forward-only 2-coil objective takes
   29 s, of which roughly 20 s is eager beam assembly that jits down to under
   a millisecond. **[measured]**
3. **The coupled operator is non-self-adjoint with an indefinite symmetric
   part whenever `k_tor ≠ k_lin`**, and that is what stops the block
   Gauss–Seidel loop from ever reaching a fixed point: it marches along a
   single direction with eigenvalue 1.0000044 while the residual stays flat
   from sweep ~7, then returns the unconverged state with no warning. The
   behaviour is independent of coil shape, attachment locality, and `k_lin`,
   but `k_tor` controls it exactly — at `k_tor = k_lin` the matrix becomes
   symmetric to machine precision, positive definite, and 4 orders of
   magnitude better conditioned. The same `K_ss` feeds the monolithic cuDSS
   path, so this is not confined to the deprioritised driver. **[measured]**

Beyond that, roughly 600 lines across the package have no caller anywhere in
`src/`, `tests/`, `examples/`, or `docs/`, the von Mises stress kernel is
written out three times, and the beam-network assembly recomputes the same
per-endpoint weights and moment arms two to four times per evaluation.

Priority ordering:

| Priority | Item | Where |
|---|---|---|
| P0 | Staggered gradient is broken; the IFT `custom_vjp` is dead | [A1](#a1--jaxgrad-through-solve_staggered-raises) |
| P0 | Coupled system has a marginal mode; BG-S never converges and says nothing | [A2](#a2--block-gaussseidel-does-not-converge-and-says-nothing) |
| P0 | `K_ss` is near-singular (cond 9.1e9) and feeds the monolithic path too | [A2c](#a2c--the-support-stiffness-block-is-near-singular) |
| P0 | `hollow_circle` / `solid_rectangle` / `solid_square` presets cannot be used | [A4](#a4--three-of-the-four-cross-section-presets-are-unusable) |
| P1 | JIT the beam side and the body-force block | [4a](#4a-jit-the-beam-side-biggest-single-win-on-cpu), [4c](#4c-jit-_body_force_at_quads) |
| P1 | `cudss_mtype_id` has two different defaults for the same key | [A5](#a5--cudss_mtype_id-has-two-conflicting-defaults) |
| P1 | Mutable default arguments mutated in place | [A6](#a6--mutable-default-arguments-are-mutated-in-place) |
| P2 | Dead code removal (~700 lines) | [Rule 1](#rule-1--does-this-need-to-exist) |
| P2 | Collapse the three von Mises implementations | [2a](#2a-von-mises--cauchy-stress-written-three-times) |
| P2 | Replace per-endpoint Python loops with `vmap` | [4b](#4b-replace-per-endpoint-python-loops-with-vmap) |
| P3 | Redundant geometry recomputation (volume 3×, surface 3×) | [2d](#2d-fe-geometry-recomputed-three-times-per-coil-per-evaluation) |
| P3 | `Support` reaches into `CoilFEM` privates for plotting/VTU | [P2](#philosophy-2--support-must-be-independent-of-coilfem) |

---

## A. Correctness issues found while applying the rules

These are not rule violations as such, but they came out of the same reading
pass and they change what the rule-driven cleanup should look like — there is
no point tidying machinery that does not run.

### A1 — `jax.grad` through `solve_staggered` raises

**[measured]** With a genuinely coupled `SupportBeams` and
`coupling='staggered'`, the forward pass works but the gradient does not:

```
=== forward staggered ===
J = 0.36798387349276307
=== grad through staggered ===
ConcretizationTypeError: Abstract tracer value encountered where concrete
value is expected: traced array with shape float64[]
The problem arose with the `float` function.
```

The failure is in the convergence check:

```321:341:src/coil_fem/coupling/drivers.py
        for k in range(max_iters):
            u_s_new, sol_list = _sweep_full(u_s)
            delta = u_s_new - u_s          # f(u_s) = T(u_s) - u_s
            res   = float(jnp.max(jnp.abs(delta)))
```

Under `jax.grad`, `u_s_new` carries tracers (it depends on the coil DOFs
through the FEM solves), so `float(...)` cannot be evaluated. The existing
test suite does not catch this because `test_staggered_fixed_point_trivial`
uses a mock support whose `solve` returns a *constant* `jnp.zeros`, which
keeps `delta` concrete.

The `custom_vjp` machinery around it is dead in three separate ways, so
fixing the `float()` alone would not give correct gradients:

- The module docstring and `solve_staggered`'s docstring both say the fixed
  point is "wrapped in `jax.lax.custom_root`". There is no `custom_root`
  anywhere in the file. What exists is `_staggered_core`, a `@jax.custom_vjp`
  **identity** applied to `u_s_star` *after* the Python loop has already run.
- The objective consumes `sol_list_by_coil`, not `u_s`. `sol_list_by_coil`
  comes straight out of the last `_sweep_full` call and never passes through
  `_staggered_core`, so the metric gradient would not see the IFT correction
  even if the loop were traceable.
- `_staggered_core_bwd` returns the GMRES solution as the cotangent with
  respect to its own input, which then *also* propagates back through the
  unrolled iteration history — double counting. The code comment
  (`drivers.py:400-405`) effectively concedes this: "the closure params …
  flow through the standard autodiff path".

Two honest options:

1. Delete `_staggered_core`, `_sweep`, and the IFT prose, and document
   `solve_staggered` as forward-only (which matches the philosophy point that
   monolithic + cuDSS is the path that matters for optimisation).
2. Rewrite the fixed point as a `lax.while_loop` with a bounded trip count and
   put a real `custom_vjp` (or `lax.custom_root`) around the *whole* solve so
   that both `u_s*` and `sol_list` are outputs of the differentiated
   primitive. This also makes the loop jittable, which is where most of the
   CPU time is going ([4a](#4a-jit-the-beam-side-biggest-single-win-on-cpu)).

### A2 — Block Gauss–Seidel does not converge, and says nothing

**[measured]** On a 2-coil, 4-beam network the loop runs all 100 sweeps:

```
BG-S sweeps: 100   J = 0.36798387349276307
```

`_run_iterations` breaks on `res < atol` and otherwise falls out of the `for`
loop and returns whatever it has. There is no warning, no diagnostic, and
`result['diagnostics']` is hard-coded to `{}` (`drivers.py:424`). A caller has
no way to know the answer is not a fixed point.

The reporting gap is the easy half to fix: emit a warning with the final
residual and populate `diagnostics` with
`{'iterations', 'residual', 'converged'}`. The harder half is why it does not
converge.

#### A2a — It is not the coil geometry, the clamp, or the stiffness

The first run used the repo's own fixtures: `_make_curves` from
`tests/test_beam_networks.py`, which is **coplanar circles** of radius 1.0 and
1.1 in the *xz*-plane, and `_uniform_clamp_fn`, which returns weight 1.0 at
*every* surface quadrature point — so each of the three beam endpoints touching
coil 0 clamps its entire 0.74 m² surface (300 % of the coil area in aggregate).
That is a soft configuration in one sense and a pathological one in another, so
I re-ran it across the axes that normally govern block Gauss–Seidel
convergence. **[measured]**, all with `max_iters = 100`:

| Coil shape | Attachment | `k_lin` | Sweeps | Residual `max|T(u)−u|` |
|---|---|---:|---:|---|
| Coplanar circles R = 1.0, 1.1 | uniform, 300 % of area | 1e8 | 100 | 9.7e-6 → 1.19e-5, flat from k≈7 |
| Coplanar circles R = 1.0, 1.1 | sigmoid ball r = 0.06, 9.4 % of area | 1e8 | 100 | 9.5e-4 → 3.34e-4, flat from k≈7 |
| Coplanar circles R = 1.0, 1.1 | sigmoid ball r = 0.06 | 1e5 | 100 | 9.5e-1 → 3.39e-1, flat from k≈7 |
| Offset, non-coplanar circles (Δy = 0.45) | sigmoid ball r = 0.06 | 1e8 | 100 | 2.113e-2 → 2.119e-2, never decreases |

So it is not caused by the coil shape (the non-coplanar case is *worse* — the
residual never decreases at all), not by the whole-surface clamp in the
fixture, and not by the coupling stiffness (the residual scales exactly as
`1/k_lin` while the iteration behaviour is identical). Non-planar,
randomly-perturbed order-3 coils were also tried but did not complete within
the review budget; see [A2d](#a2d--higher-order-curves-are-pathologically-slow).

This is the opposite of the usual expectation that irregular coil shapes
destabilise the partitioned scheme. It already fails on two concentric
circles, which is about the most benign geometry available.

#### A2b — The iteration marches along a mode with eigenvalue 1.0000044

Instrumenting the sweep (recording `u_s` on the way into `compute_attach` and
on the way out of `support.solve`) shows the signature clearly. Offset-circle
case, `k_lin = 1e8`, `k_tor = 1e4`: **[measured]**

```
   k    omega_effective   ||T(u)-u||     ||u_s||
   0         1.000000     3.378882e-02   0.000000e+00
   1         0.100000     3.451686e-02   3.378882e-02
   ...
  14         0.100000     3.456616e-02   7.852101e-02
  15         0.100000     3.456624e-02   8.196935e-02
  16         0.180800     3.456606e-02   8.541803e-02
  ...
  20         2.000000     3.453636e-02   1.772595e-01
  21         2.000000     3.451431e-02   2.462480e-01
  22         0.100000     3.452460e-02   3.151921e-01
  ...
  28         0.100000     3.452555e-02   3.358820e-01

cos angle between successive late increments: 0.999999989
ratio of successive late increment norms:     1.000004660
```

Three things to read off this:

- `‖T(u) − u‖` is constant to three digits for 29 sweeps. It never decays.
- `‖u_s‖` grows with a per-sweep increment that is constant to four digits, and
  the increment *direction* is fixed: successive increments have
  `cos = 0.999999989` with a norm ratio of `1.0000047`.
- The effective relaxation factor sits on the clamp floor `0.1` most of the
  time, with excursions to the ceiling `2.0`.

A fixed direction reproduced with ratio ≈ 1 means the BG-S map `T(u) = Au + b`
has an eigenvalue of essentially exactly 1 (marginally above it), and the
residual has a component in that eigenspace. There is therefore no fixed point
to converge to along that direction, and the iteration translates along it
forever. Aitken cannot rescue this: for a unit eigenvalue no scalar `ω`
produces a correction in that direction, and the clamp

```333:333:src/coil_fem/coupling/drivers.py
                    omega = max(0.1, min(2.0, omega))
```

separately forbids the strong under-relaxation (`ω ≪ 0.1`) that a marginally
stable coupling would need even if the mode were merely stiff rather than
singular. That clamp is worth revisiting on its own merits.

The one stiffness that *does* control the mode is `k_tor`; see
[A2c](#a2c--the-support-stiffness-block-is-near-singular).

#### A2c — The support stiffness block is near-singular

Assembling `K_ss` and the coupling blocks directly and sweeping `k_tor` with
`k_lin = 1e8` fixed **[measured]**:

| `k_tor` | `‖K_ss − K_ssᵀ‖/‖K_ss‖` | `‖K_cs − K_scᵀ‖/‖K_cs‖` | negative eigenvalues of `sym(K_ss)` | `σ_min(K_ss)` | cond `K_ss` |
|---:|---|---|---:|---|---:|
| 1e4 (fixture) | 1.249e-03 | 2.726e-02 | 4 / 48 | 1.696e-02 | 9.10e9 |
| 1e6 | 1.236e-03 | 2.699e-02 | 4 / 48 | 1.695e+00 | 9.10e7 |
| 1e8 (= `k_lin`) | **1.7e-17** | **0.0 exactly** | **0 / 48** | 1.611e+02 | 9.58e5 |

`k_tor` is the knob. `σ_min(K_ss)` tracks it linearly over four decades, and at
`k_tor = k_lin` the four negative eigenvalues of the symmetric part vanish, the
matrix becomes symmetric to machine precision, and the condition number drops
by four orders of magnitude. So the weak subspace is exactly the rotational
DOFs, whose only external stiffness is `k_tor`.

The reason is structural rather than a tuning accident.
`_spring_stiffness_contributions` writes the translation–rotation block as
`−k_lin Σ w [r]×` and the torque–translation block as `+k_tor Σ w [r]×`
(`beam_network.py:1797-1803`). Since `[r]×ᵀ = −[r]×`, the transpose of the
first is `+k_lin Σ w [r]×`, which equals the second only when
`k_tor = k_lin`. `coupling_values` has the same structure:
`(blk_r_cs)ᵀ = −k_lin · skew_eff · Q` while `blk_r_sc = −k_tor · skew_eff · Q`
(`beam_network.py:1541-1547`). Translation–translation blocks are symmetric
unconditionally, and the bare beam `K_global = Γ K_local Γᵀ` is symmetric by
congruence, so **`k_tor` is the only source of asymmetry in the whole merged
system**, and it enters both the diagonal `K_ss` and the off-diagonal coupling.

A non-self-adjoint operator whose symmetric part is indefinite is exactly the
setting in which a partitioned Dirichlet–Neumann iteration has a growing mode,
which is what [A2b](#a2b--the-iteration-marches-along-a-mode-with-eigenvalue-10000044)
observes. This looks like a modelling choice with numerical consequences rather
than a coding defect: the torque law `τ = k_tor Σ w r × Δu` with an independent
`k_tor` is what breaks self-adjointness. A single foundation modulus (a genuine
distributed spring bed, `k_tor ≡ k_lin`) would make the entire merged system
symmetric positive definite for free — worth deciding deliberately, since
`CoilFEM` already enforces `winkler_k == k_lin` but places no constraint on
`k_tor`.

A condition number of 9 × 10⁹ on a 48 × 48 matrix has two further direct
consequences:

1. `SupportBeams.solve` densifies this block and factors it with
   `lineax.LU()` (`beam_network.py:2024-2037`), losing roughly 10 of 16
   digits.
2. The **same** `K_ss` values go into the merged monolithic matrix via
   `make_merged_solve` — so this is not confined to the staggered path that
   the philosophy deprioritises. It sits in the cuDSS path too.

Two further defects in the same rotational subspace, which compound the above:

- **The CF foundation endpoint provides no rotational grounding at all.**
  `_endpoint_weights_and_r` sets `r_fnd = x_foundation[i][j] - geom['x_end'][b]`
  where `geom['x_end'][b]` *is* `x_foundation[i][j]` (the code says so at
  `beam_network.py:1704`), so `r_fnd ≡ 0`, hence `skew_sum = skew2_sum = 0`,
  hence `K_tr = K_rt = K_rr = 0` at the foundation node. The foundation
  contributes translational stiffness only, and `_assemble_rhs` explicitly
  does nothing for it (`pass`, `beam_network.py:1952`). The foundation-side
  rotation DOFs are then restrained only through the bare beam's own bending
  and torsion — so even setting `k_tor = k_lin` leaves the foundation nodes
  with no rotational spring, because `k_tor` multiplies a zero moment arm
  there.
- **A units mismatch between the coil side and the foundation side.** After the
  working-tree JxW change, every coil-side endpoint contributes
  `K_tt += k_lin · Σ(w · JxW) · I₃` — an *area*-weighted sum — while the
  foundation endpoint still contributes `K_tt += k_lin · 1.0 · I₃` with a
  dimensionless `w_sum = 1.0` (`beam_network.py:1778-1781`). With
  `k_lin = 1e8` N/m³ the coil side is `≈ 7 × 10⁶` N/m and the foundation side
  is `10⁸` N/m — 14× stiffer, for no physical reason. The JxW change updated
  the coil branch and left the foundation branch alone.

A cheap diagnostic that would have surfaced this: `merged_solve` already gets
the matrix inertia back from cuDSS and throws it away
(`sol_flat, _inertia = solver_K(f_merged, csr_values)`, `drivers.py:559`).
Surfacing it, or asserting on `min(svd(K_ss))` in a test, would catch a
zero-energy mode at assembly time instead of as a silent non-convergence 100
sweeps later.

#### A2d — Higher-order curves are pathologically slow

Also **[measured]**, and unexplained: with everything else held fixed, swapping
the order-1 circles for randomly perturbed order-3 curves takes a case that
runs in 3–8 s to over 10 minutes without completing (three separate attempts,
including one at 20 minutes). The same happened for order-2 perturbations. The
per-sweep work should not depend on the Fourier order — the curve is evaluated
once per objective call, not once per sweep — so the likely culprits are
distorted or inverted tetrahedra making the umfpack solve pathological, or
repeated re-lowering of the RMF `lax.scan` that
`framed_curve_jax.py:226-238` warns about. Worth reproducing and profiling,
because randomly-shaped coils are the realistic optimisation input.

Note that non-convergence also means every staggered solve currently pays the
full 100 sweeps, which is why the timing in
[Rule 4](#rule-4--jit-opportunities) looks the way it does.

### A3 — The returned solutions do not correspond to the returned `u_s`

```335:342:src/coil_fem/coupling/drivers.py
            u_s = u_s + omega * delta
            last_sol_list = sol_list

            if res < atol:
                break
            delta_prev = delta

        return u_s, last_sol_list
```

`sol_list` was produced by `_sweep_full(u_s)` *before* the Aitken update, so
the returned `u_s` and `last_sol_list` are one relaxation step out of sync.
On a converged run the mismatch is below `atol`; combined with
[A2](#a2--block-gaussseidel-does-not-converge-and-says-nothing) it is not.

### A4 — Three of the four cross-section presets are unusable

`CoilSupportBeams` resolves the attachment function by string concatenation:

```517:522:src/coil_fem/simsopt/optimizables.py
        if attachment_type == 'direct':
            attachment_fn = fetch_attr(cross_section_type + '_attachment', cross_section_fns)
        elif attachment_type == 'wrap':
            attachment_fn = cross_section_fns.wrap_attachment
            cross_section_option_keys += cross_section_fns.wrap_option_keys
```

`attachment_type` defaults to `'direct'`, but `presets/cross_section_fns.py`
only defines `solid_circle_attachment` and the alias
`hollow_circle_attachment`. There is no `solid_rectangle_attachment` and no
`solid_square_attachment`, so `cross_section_type='solid_rectangle'` or
`'solid_square'` raises `ValueError` from `fetch_attr` at construction.

The `hollow_circle` alias is worse because it fails later and less obviously:

```298:298:src/coil_fem/presets/cross_section_fns.py
hollow_circle_attachment = solid_circle_attachment
```

`solid_circle_attachment` reads `dofs['r_beam']`, but
`hollow_circle_dof_keys = ('r_1_beam', 'r_2_beam')`, so
`_clamp_weights_for_spec` never puts `r_beam` into `beam_dofs` and the call
raises `KeyError` during the first assembly.

Separately, an `attachment_type` that is neither `'direct'` nor `'wrap'`
leaves `attachment_fn` unbound and raises `UnboundLocalError` rather than a
useful message. Add an `else: raise ValueError(...)`.

### A5 — `cudss_mtype_id` has two conflicting defaults

`build_monolithic_static` uses `0` (general), correct for the merged system
since `K_ss` is explicitly documented as non-symmetric:

```439:439:src/coil_fem/coil_fem.py
            mtype_id  = int(self.problem_options.get('cudss_mtype_id', 0))
```

`build_fwd_pred` uses `1` (symmetric) for the same key:

```74:74:src/coil_fem/solvers/__init__.py
            mtype_id=int(problem_options.get('cudss_mtype_id', 1)),
```

Both read the *same* `problem_options` dict, and — this is the point — **each
default is individually correct for its own matrix**, so there is no single
value that can serve both.

`cudss_mtype_id` is passed through spineax to cuDSS's `cudssMatrixType_t`
(the if-chain is at `spineax/cudss/single_solve.cpp:157-174`) and declares
which structural property the solver may exploit. It is a promise, not a
check — cuDSS never validates the matrix against it:

| id | cuDSS type | Mathematical requirement | Factorisation |
|---:|---|---|---|
| 0 | `GENERAL` | none | LU, partial pivoting |
| 1 | `SYMMETRIC` | `A = Aᵀ` | LDLᵀ, handles indefinite |
| 2 | `HERMITIAN` | `A = Aᴴ` | LDLᴴ |
| 3 | `SPD` | `A = Aᵀ` and `xᵀAx > 0 ∀x ≠ 0` | Cholesky, no pivoting |
| 4 | `HPD` | `A = Aᴴ` and positive definite | Cholesky, complex |

coil-fem is real `float64` throughout, so 2 and 4 are unreachable in practice
(`Aᴴ = Aᵀ` for real `A`, making them equivalent to 1 and 3). The 1-vs-3
distinction is definiteness, not symmetry.

What the two matrices actually are **[measured]**:

- **Single-coil `K_cc`** is symmetric to machine precision,
  `‖K − Kᵀ‖/‖K‖ = 6.0e-17`, and structurally so: the elasticity tangent
  `∫ C ε(u):ε(v) dV` inherits the major symmetry `C_ijkl = C_klij`, the
  Winkler term `∫ k w u·v dS` is a surface mass matrix, and
  `apply_symmetric_dirichlet` zeros rows *and* columns and folds the columns
  into the RHS — which is what its name is about. Definiteness depends on the
  clamp: with `w = 1` everywhere the smallest eigenvalue is `+1.12e6` against
  a largest of `2.07e11` (genuinely SPD, cond ≈ 1.9e5), but with `w = 1` on
  only 10 % of the surface quadrature points five eigenvalues fall within
  `1e-6` of zero relative to the largest and the condition number rises to
  ≈ 6.6e9. Those five are the rigid-body modes a partial clamp does not pin.
  So `1` is the sound default and `3` is not safe in general, because the
  sigmoid clamps used in practice do underflow to exactly zero away from the
  clamp.
- **The merged monolithic matrix** is symmetric **iff `k_tor = k_lin`**, and
  is otherwise both non-symmetric and indefinite in its symmetric part; see
  the sweep in [A2c](#a2c--the-support-stiffness-block-is-near-singular). With
  the fixture's `k_tor = 1e4, k_lin = 1e8` the relative asymmetry is `1.2e-3`
  in `K_ss` and `2.7e-2` between `K_cs` and `K_scᵀ`, so `0` is required.

The hazard is therefore worse than a mistuned option. With `mview_id`
hard-coded to `0` (`CUDSS_MVIEW_FULL`, `coil_fem.py:441`) you hand cuDSS the
complete matrix and then tell it which triangle it may ignore; declaring
`SYMMETRIC` for the merged system yields the solution of a symmetrised matrix
with no error raised. Use two distinct option keys
(`cudss_mtype_id_coil` / `cudss_mtype_id_merged`), or derive the merged value
from `support.k_tor == support.k_lin` and assert it at construction. If
`k_tor ≡ k_lin` were adopted as the model (see
[A2c](#a2c--the-support-stiffness-block-is-near-singular)), `1` would become
valid for the merged matrix too, and cheaper than `0`.

### A6 — Mutable default arguments are mutated in place

```483:491:src/coil_fem/simsopt/optimizables.py
        beam_options={},
        ...
        fixed_clamp_options={'enabled': False},
```

and then:

```526:526:src/coil_fem/simsopt/optimizables.py
        beam_options.setdefault('sigmoid_eps', 0.1)
```

The default `{}` is shared across every call, so the first
`CoilSupportBeams()` built without an explicit `beam_options` poisons the
default for the rest of the process. Even with an explicit dict, `setdefault`
mutates the caller's object. Use `beam_options=None` plus
`beam_options = dict(beam_options or {})`.

### A7 — Latent trap: cuDSS default is the host-syncing Newton loop

`CoilFEMObjective` enables `jax.jit` precisely when `solver == 'cudss'`:

```166:176:src/coil_fem/simsopt/objectives.py
        _use_jit = (
            problem_options is not None
            and problem_options.get('solver', 'umfpack') == 'cudss'
        )
```

but `build_fwd_pred` passes `linear=bool(problem_options.get('cudss_linear', False))`,
so the default `CuDSSNewtonSolver.newton_loop` takes the iterative branch,
which calls `float(jnp.linalg.norm(res_vec_bc))` — the same host sync that
breaks tracing in [A1](#a1--jaxgrad-through-solve_staggered-raises).
`cudss_linear` is not set anywhere in the repo, tests, or examples.

This only bites on `coupling='staggered'` with `solver='cudss'`, or the
uncoupled cuDSS path (the monolithic path never calls `fwd_pred`), but it is a
sharp edge. `LinearElasticity3D` is linear by construction — default
`linear=True` when the problem is a `LinearElasticity3D`, or drop the
non-linear branch entirely (see [3f](#3f-the-non-linear-newton-branch-is-ballast)).

### A8 — Unreachable default-from-support branch

```298:311:src/coil_fem/coil_fem.py
        winkler_k = float(self.problem_options['winkler_k'])
        ...
        if self.support.is_coupled and hasattr(self.support, 'k_lin'):
            if self.problem_options.get('winkler_k') is None:
                # Default winkler_k from support
                winkler_k = float(self.support.k_lin)
```

`_broadcast_problem_options` already raises if `'winkler_k'` is absent, and
line 298 does `float(...)` on it before this branch is reached — so
`winkler_k is None` either never happens or has already raised `TypeError`.
Either delete the branch or move the default-from-support resolution above
line 298 and relax the hard requirement in `_broadcast_problem_options`.

### A9 — `disk` meshes are accepted but cannot be solved

`_broadcast_mesh_opts` validates and accepts `shape='disk'`
(`coil_fem.py:94-106`), and `CoilMeshDisk` builds fine, but `CoilMeshDisk`
never overrides `_compute_uv_quad`, so `mesh.uv_quad` stays `None`, and
`B_self_quadrature` raises `NotImplementedError` for `shape == 'disk'`
regardless. Any `CoilFEM` built with a disk cross-section fails on the first
`run()`/`objective()`. Reject `'disk'` at construction with the
`NotImplementedError` message that `B_self_quadrature` already has, so the
failure happens where the user can act on it.

### A10 — `k_lin` units are documented inconsistently

`SupportBeams` documents `k_lin : Translational spring stiffness [N/m²]`
(`beam_network.py:133`) and `CoilSupportBeams` repeats it
(`optimizables.py:433`). But `CoilFEM.__init__` enforces
`winkler_k == support.k_lin` and documents `winkler_k` as N/m³, and the
dimensional analysis agrees: `K_tt = k_lin · Σ(w·JxW) · I₃` where
`Σ(w·JxW)` is an area, so `k_lin·m² = N/m ⟹ k_lin = N/m³`. Fix the two
docstrings.

---

## Rule 1 — Does this need to exist?

### 1a. Dead code: no caller in `src/`, `tests/`, `examples/`, or `docs/`

| Symbol | Location | Note |
|---|---|---|
| `ElasticPipeline.solve_residual` | `pipelines.py:168-192` | Superseded by `assemble_coo`, which returns `load` as a by-product and says so in its own docstring |
| `ElasticPipeline.attachment_displacement` | `pipelines.py:241-253` | Drivers use `u_at_surface_quads` |
| `ElasticPipeline.coo` | `pipelines.py:255-279` | Docstring literally says "Plan B preparation"; `assemble_coo` supersedes it |
| `meshing.validate_mesh` | `meshing.py:197-248` | Plus 8 lines of JIT-caveat comments above it |
| `CoilMesh.to_vtu` | `meshing.py:787-798` | Hard-codes its own `TET4→tetra` map next to the `meshio_cell_type` property that already does this |
| `CoilMesh.mesh_longest_edge_volume_ratio` | `meshing.py:837-876` | |
| `CoilMesh.mesh_type` | `meshing.py:755-758` | "Alias for ele_type for backward compatibility"; the external `'mesh_type'` hits are all dict keys, not this property |
| `CurveXYZFourierJAX.kappa` / `.torsion` / `.frenet_frame` | `curve_jax.py:144-216` | `gammadashdashdash` exists only to serve `torsion` |
| `FramedCurveJAX.binorm` / `.torsion` | `framed_curve_jax.py:548-554` | Aliases of `frame_binormal_curvature` / `frame_torsion` |
| `problems.dirichlet_bc` | `linear_elasticity.py:83-123` | Exported in `problems.__all__`; the only references are its own docstring examples. `LinearElasticity3D` raises if `location_fns` is passed alongside `winkler_k_scalar` |
| `HeatConduction3D` | `problems/heat_conduction.py` | Whole module is a class that raises on `__init__`; re-exported in `problems.__all__` |
| `CoilFEM.n_total` | `coil_fem.py:274` | Assigned, never read |
| `solve_staggered(..., options=...)` | `drivers.py:123` | `CoilFEM._solve_all` never passes it, so `max_iters`, `atol`, `aitken`, `gmres_maxiter`, `gmres_tol` are all unreachable from the public API — directly relevant to [A2](#a2--block-gaussseidel-does-not-converge-and-says-nothing) |
| `'amgx'` in `_VALID_SOLVERS` | `coil_fem.py:110` | Accepted by validation; `build_fwd_pred` would hand `{'amgx_solver': {}}` to jax-fem |
| `_HAS_SPINEAX` | `solvers/cudss.py:36` | Computed via `importlib.util.find_spec` and never read |
| Commented-out plot block | `coil_fem.py:1356-1364` | |

`ThermoElasticPipeline` (`pipelines.py:282-295`) is a near-miss: it is
reachable via `physics_options={'type': 'thermoelastic'}`, constructs
successfully, and then raises on `solve`. Its `solve` signature also omits
`support_attach`, so on the coupled path it raises `TypeError` before it can
raise the intended `NotImplementedError`. Either wire it up or drop it and the
`physics_options` plumbing with it; a stub that constructs but cannot run is
worse than no stub.

### 1b. Placeholder implementations that the architecture no longer uses

**`displacement_at`.** `Support.displacement_at` returns zeros;
`SupportBeams.displacement_at` also returns zeros with a
`TODO(driver-integration)` note. Neither driver calls it — they use
`compute_attach`, which is the method that actually does the
rigid-body-displacement interpolation. AGENTS.md still lists
`displacement_at` as a required abstract method and
`docs/developers/support_structure.rst` devotes a step to implementing it.
Delete the method, or delete `compute_attach` and make `displacement_at` do
the work; having both, with the documented one being the dead one, is the
worst arrangement.

**`coupling_terms`.** `Support.coupling_terms` (48 lines, `supports.py:527-578`)
and `SupportBeams.coupling_terms` (80 lines, `beam_network.py:1556-1635`)
have no caller in `src/`. `make_merged_solve` calls `coupling_pattern` and
`coupling_values` separately, which is the right split (static indices once at
construction, traced values every evaluation). `coupling_terms` just glues them
back together and is kept alive only by three tests. Note that
`SupportBeams.coupling_terms` also forgets to forward `geom`, so calling it
recomputes the beam geometry from scratch. Either delete it and have the tests
call the two halves, or keep it as a documented test/debug convenience — but
its physics-convention docstring (the clearest explanation of the `K_cs`/`K_sc`
sign conventions in the codebase) should move to `coupling_values`, which is
the method that implements it.

### 1c. Base-class contracts that do not match the subclass

**`Support.coo(self)`** takes no arguments and raises `NotImplementedError`.
`SupportBeams.coo(self, curves_jax, support_dofs, surface_pts_by_coil=None,
geom=None, *, jxw_by_coil)` takes five. Drivers call the five-argument form,
so a hypothetical uncoupled support reaching that code path gets `TypeError`,
not the intended `NotImplementedError`. Give the base method the real
signature.

**`Support.solve`** returns `{}` while every caller does
`support.solve(inputs)['u_s']` — a `KeyError`, not a graceful no-op. It is
only safe because it is never reached when `is_coupled=False`. Either return
`{'u_s': jnp.zeros(0)}` or make it raise.

### 1d. Single-use wrappers that add a hop without adding meaning

| Wrapper | Location | Suggestion |
|---|---|---|
| `_build_metric_fn` | `coil_fem.py:65-71` | A dict lookup plus an error message, called once in a comprehension. Inline into `objective` |
| `metrics._resolve_shape_grads` / `_resolve_JxW` | `metrics.py:50-61` | One-line `if x is not None` fallbacks. `objective` always passes both |
| `utils.fetch_attr` | `utils.py:3-8` | Reimplements `getattr` and downgrades `AttributeError` to `ValueError`. See [3a](#3a-stdlib-and-jax-built-ins) |
| `meshing.rectangle_sweep` / `disk_sweep` | `meshing.py:503-551` | Self-described "backward-compatible shim"s that call the constructor. Only used in tests and examples — update those and delete |
| `_scatter_block_diagonal` | `beam_network.py:1818-1835` | Returns `(self._coo_I, self._coo_J, K_beam.reshape(-1))`; one call site |
| `_bool_to_sign` | `cross_section_fns.py:48-52` | Five lines for `1 if a else -1`; one call site |
| `CoilFEM.plot_support` / `save_support_vtu` | `coil_fem.py:1185-1286` | Pure forwards to `Support` methods that immediately reach back into `CoilFEM` privates. See [Philosophy 2](#philosophy-2--support-must-be-independent-of-coilfem) |

`_broadcast_problem_options` (`coil_fem.py:113`) does not broadcast anything —
it validates one dict and fills defaults. Rename to
`_validate_problem_options`.

`CoilFEM.meshes` (`coil_fem.py:1124-1132`) is described as a
"backward-compatibility shim", but it is now read 8× inside `coil_fem.py`
itself, including inside `_solve_all`'s per-coil loop, and it rebuilds a list
comprehension on every access. Either keep it and cache it in `__init__`, or
change the internal call sites to `self.pipelines[i].mesh` and keep the
property for external users.

`presets/__init__.py` is a two-line docstring with a typo ("constantas") that
re-exports nothing, while AGENTS.md describes it as the presets namespace.

### 1e. Copy-pasted docstrings

Three `SupportBeams` properties carry the docstring of a fourth:

```392:405:src/coil_fem/coupling/beam_network.py
    @property
    def nfp(self) -> bool:
        """``True`` — beams have their own DOFs coupled to coil surface nodes."""
        return self._nfp

    @property
    def beam_options(self) -> bool:
        """``True`` — beams have their own DOFs coupled to coil surface nodes."""
        return self._beam_options

    @property
    def stellsym(self) -> bool:
        """``True`` — beams have their own DOFs coupled to coil surface nodes."""
        return self._stellsym
```

`nfp` and `beam_options` are also annotated `-> bool`.

---

## Rule 2 — Duplicates already in this codebase

### 2a. Von Mises / Cauchy stress written three times

The same constitutive law appears in:

- `metrics.cauchy_stress_small_strain` + `metrics.von_mises_on_quadrature`
  (`metrics.py:16-133`)
- `LinearElasticity3D.von_mises_stress`, via the inline `vm_at_point`
  (`linear_elasticity.py:904-912`)
- `LinearElasticity3D.get_tensor_map`'s `stress` closure
  (`linear_elasticity.py:668-672`)

and `CoilFEM.strain_energy_density` (`coil_fem.py:930-957`) recomputes
`eps`/`eps_m` a fourth time before delegating to
`cauchy_stress_small_strain`, which recomputes them again internally.

This is not only redundant, it is a live inconsistency: `CoilFEM.run` uses
`problem.von_mises_stress` (`coil_fem.py:846`) while `CoilFEM.objective` uses
`metrics.von_mises_on_quadrature` (via the registry). The two differ in
exactly the way the `objective` code comment warns about — `von_mises_stress`
reads the mutated `self.shape_grads` left behind by the last `set_params`,
which is the state leak `objective` recomputes geometry to avoid:

```1056:1064:src/coil_fem/coil_fem.py
            # Recompute FE geometry OUTSIDE the ad_wrapper custom_vjp scope
            # so that JAX can differentiate through shape_grads and JxW via
            # standard AD.  Reading problem.shape_grads (set as a side effect
            # inside the custom_vjp forward) would leak a traced value across
            # the custom_vjp boundary and produce NaN gradients.
```

Keep `metrics.py` as the single owner of the constitutive kernel;
`get_tensor_map` should call `cauchy_stress_small_strain`, and
`von_mises_stress` / `strain_tensors` should either take `shape_grads`
explicitly (as `strain_tensors` already optionally does) or be deleted in
favour of the `metrics` functions.

`displacement_gradient_at_quads` is likewise open-coded in
`von_mises_stress` (`linear_elasticity.py:897-899`) and `strain_tensors`
(`linear_elasticity.py:958-960`), identically to
`metrics.displacement_gradient_at_quads`.

### 2b. Rotation and symmetry primitives written twice

- **Rodrigues rotation.** `beam_network._rodrigues` (`beam_network.py:38-59`)
  and `framed_curve_jax._angle_axis_rotation_matrix`
  (`framed_curve_jax.py:97-116`) are the same formula. Keep one, in `geo/`.
- **Stellarator symmetry.** `geo/symmetries` has `_rotate_points_z` and
  `_flip_points` for coil expansion; `SupportBeams.__init__` independently
  builds `diag(1,-1,-1)` and the `2π/nfp` z-rotation into `self._tfm_Q`
  (`beam_network.py:283-294`). Per philosophy point 6, the `Q` matrices belong
  in `geo/symmetries.py` (e.g. `stellsym_transform(tag, nfp) -> (3, 3)`), with
  `beam_network` importing them. That also makes it checkable that the beam
  network's `'flip_half' = rot @ flip` convention matches the coil expansion's
  `flip(rotate(x))` ordering.

### 2c. `_sweep` and `_sweep_full` are the same function

`drivers._sweep` (`drivers.py:230-266`) and `_sweep_full`
(`drivers.py:268-308`) differ only in whether `sol['sol_list']` is appended to
a second list — 35 duplicated lines including the `geom_kw` dance and the
`support_inputs` dict. `_sweep` exists only for `_staggered_core_bwd`, which
is dead ([A1](#a1--jaxgrad-through-solve_staggered-raises)). Delete `_sweep`,
or define it as `lambda u: _sweep_full(u)[0]`.

### 2d. FE geometry recomputed three times per coil per evaluation

`recompute_fe_geometry` is called on the same `pts_i`:

1. in `_body_force_at_quads` (`coil_fem.py:637`), which discards `sg`, `JxW`,
   and `v_grads_JxW` and keeps only `pqp`;
2. inside `LinearElasticity3D.set_params` (`linear_elasticity.py:772`) during
   the solve;
3. in `objective` (`coil_fem.py:1061`) for `sg_ext` / `jxw_ext`.

**[measured]** ~1.0 ms per call on a 768-cell TET4 mesh, so ~2 ms/coil wasted,
scaling linearly with mesh size. The surface geometry is worse: the
`physical_coos → selected_coos → jacobian → det_J/inv_J` chain is written out
independently in `surface_quad_points` (`linear_elasticity.py:568-575`),
`surface_jxw` (`linear_elasticity.py:605-621`), and the surface block of
`set_params` (`linear_elasticity.py:783-812`) — **[measured]** ~1.0 ms and
~1.5 ms respectively. In `make_merged_solve._assemble_merged_values` all three
run on the same `pts`, and `_bwd` runs the whole set twice more.

Suggestion: one `_fe_geometry(points)` and one `_surface_geometry(points)`
returning a small dataclass/dict, computed once per `(coil, points)` and
threaded through. This is the priority (cuDSS/monolithic) path, so it is worth
doing carefully.

### 2e. `curves_jax` rebuilt in eight places

The same three-line comprehension

```python
[CurveXYZFourierJAX(base.quadpoints, d, base.order)
 for base, d in zip(self.base_curves_jax, base_curves_dofs)]
```

appears in `CoilFEM._expand_geometry` (in loop form), `CoilFEM._solve_all`,
`CoilFEM.save_run_vtu`, `Support.plot_support`, `Support.save_support_vtu`,
`SupportBeams.plot_support`, `SupportBeams.save_support_vtu`, and
`drivers.make_merged_solve._make_curves`. Two of them (`supports.py:222`,
`supports.py:330`) import `CurveXYZFourierJAX` locally under an alias to do it.
Add `CoilFEM.curves_from_dofs(base_curves_dofs)` and have the visualisation
helpers accept the already-built list.

### 2f. Surface-weight scatter duplicated three times

The `pts_i → surf_idx → weights_surf → weight_full[surf_idx] = weights_surf`
block is byte-similar in `CoilFEM.save_run_vtu` (`coil_fem.py:1526-1537`),
`Support.save_support_vtu` (`supports.py:346-356`), and
`Support.plot_support` (`supports.py:239-249`).

### 2g. `gamma3` resolution duplicated four times

```python
gamma3 = geom['gamma3'] if 'gamma3' in geom else self._direction_cosine_matrices(geom, dofs)
```

appears verbatim in `compute_weights` (`beam_network.py:1054`),
`compute_attach` (`:1330`), `coupling_values` (`:1506`), and `coo` (`:1887`).
Since `geometry()` is the only public producer and always sets `'gamma3'`,
either make the key mandatory or add one private `_gamma3(geom, dofs)`.

### 2h. Endpoint weights and moment arms recomputed 2–4× per assembly

`_endpoint_specs(geom, gamma3)` is rebuilt, and `_clamp_weights_for_spec` is
re-run for every endpoint, in four separate methods: `compute_weights`,
`compute_attach`, `coupling_values`, and `_endpoint_weights_and_r`. In the
monolithic path a single `_assemble_merged_values` call runs
`_endpoint_weights_and_r` (inside `coo`) *and* `coupling_values`, so every
`(w_k, r_k)` pair is computed twice on the same surface points; in the
staggered path `compute_weights`, `compute_attach`, and the two inside
`solve` make it four times per sweep. Compute the endpoint quantities once per
`geometry()` call and pass them through the same way `geom` already is.

### 2i. Smaller duplications

- `CoilSupportFixed._clamp_fn` (`optimizables.py:314-321`) and
  `CoilSupportBeams._clamp_fn` (`optimizables.py:776-783`) are byte-identical.
- `MonolithicStatic` stores `curve_qps` / `curve_orders`, duplicating
  `CoilFEM.base_curves_jax[i].quadpoints` / `.order`, and
  `surface_node_indices_by_coil`, duplicating `pipeline.surface_node_indices`.
- `CoilFEM.plot`'s boundary-face extraction (`coil_fem.py:1386-1403`) reruns
  the exterior-face detection that `LinearElasticity3D.custom_init` already
  did and stored in `boundary_inds_list` (`linear_elasticity.py:340-350`).
- `beam_network._check_beam_counts` and `coil_fem._broadcast_mesh_opts` both
  hand-roll "scalar or length-N sequence" broadcasting.

---

## Rule 3 — Replaceable by stdlib, a native feature, or an installed dependency

### 3a. Stdlib and JAX built-ins

| Current | Replace with |
|---|---|
| `_gamma_12` — manual 12×12 from four 3×3 blocks (`beam_network.py:79-99`) | `jax.scipy.linalg.block_diag(g3, g3, g3, g3)`. Better still, skip forming it: `K_glob = Γ₁₂ K Γ₁₂ᵀ` on a block-diagonal `Γ₁₂` is `einsum` on the `(4,3,4,3)` reshape of `K`, which avoids a 12×12 matmul pair per beam |
| `utils.fetch_attr` (`utils.py:3-8`) | `getattr(module, name)` — already raises `AttributeError` with the module and attribute name in the message |
| `_build_static_ij`'s `np.ones((n,12,12), dtype=int)` multiplies (`beam_network.py:380-383`) | `np.broadcast_to` — the current form allocates two `n×144` integer arrays purely to achieve broadcasting |
| `jax.nn.logsumexp` (`metrics.py:227`) | `jax.scipy.special.logsumexp` — the canonical location |
| `jax.vmap(jax.vmap(_skew))` (`beam_network.py:1530`) | A single `jnp.einsum` against a constant Levi-Civita tensor |

### 3b. Local imports of stdlib and core dependencies

AGENTS.md reserves lazy imports for *optional heavy* dependencies. These are
neither:

- `import warnings as _warnings`, `import dataclasses as _dc` inside
  `CoilFEM.build_monolithic_static` (`coil_fem.py:379`, `:468`)
- `import os`, `import numpy as onp`, `import meshio` inside
  `CoilFEM.save_run_vtu`, `CoilFEM._write_coil_vtu`, `CoilFEM.plot`,
  `Support.plot_support`, `Support.save_support_vtu`,
  `SupportBeams.plot_support`, `SupportBeams.save_support_vtu`
- `import interpax` inside `CoilFEM._body_force_at_quads` and
  `B_self_quadrature` — `interpax` is a hard dependency in `pyproject.toml`
- `from .problems import recompute_fe_geometry` inside
  `CoilFEM.compute_strain_tensors` (already imported at module level,
  `coil_fem.py:31`)

`matplotlib` and `simsopt.geo.plotting` are legitimately lazy (matplotlib is a
`dev` extra); keep those.

### 3c. `meshio` already knows how to write these files

`CoilMesh.to_vtu` (dead, see [1a](#1a-dead-code-no-caller-in-src-tests-examples-or-docs))
hard-codes a second `TET4 → tetra` map next to `meshio_cell_type`, and
`CoilFEM._write_coil_vtu` re-implements a thin `meshio.Mesh(...).write(...)`
wrapper that `SupportBeams.save_support_vtu` then bypasses to call `meshio`
directly for the beam lines. One helper, used everywhere, would do.

### 3d. Unused imports and locals (pyflakes)

```
coil_fem.py:34      '.coupling.SupportBeams' imported but unused
coil_fem.py:381     local 'n_base' assigned but never used
coil_fem.py:1478    'meshio' imported but unused (imported for a side effect that meshio does not have)
meshing.py:178-179  locals 'span1'/'span2' assigned but never used
meshing.py:367      local 'n_mid' assigned but never used
pipelines.py:17     '.problems.itc_strain' imported but unused
linear_elasticity.py:494  local 'n_face_nodes' assigned but never used
drivers.py:25       'warnings' imported but unused
drivers.py:28       'numpy as onp' imported but unused
drivers.py:35-36    TYPE_CHECKING imports 'ElasticPipeline'/'Support' unused (signatures use bare `list`/untyped)
beam_network.py:1770 local 'w' assigned but never used
optimizables.py:502 local 'base_curves_jax' assigned but never used
cross_section_fns.py:42 'math' imported but unused
solvers/cudss.py:56 TYPE_CHECKING import 'LinearElasticity3D' unused
```

`beam_network.py:1770` is worth a second look: `w = ep['w']` is dead because
the non-foundation branch re-reads `ep['w'] * ep['jxw']`, and the foundation
branch hard-codes `w_sum = 1.0` — so a foundation endpoint silently ignores
its own `'w'`. That is intentional (foundation `w` is always the scalar `1.0`)
but the dead local makes it look accidental.

`meshing.py:180-181` also has two commented-out `mu_log`/`nu_log` lines that
`span1`/`span2` were computed for.

### 3e. `dataclasses` for the endpoint spec dicts

`_endpoint_specs` returns a list of 8-key dicts with string keys accessed as
`spec['coil_origin']`, `spec['j_local']`, `spec['sign_x']`, … across four
methods. A frozen dataclass (or a `NamedTuple`) gives attribute access, typo
protection, and — more usefully — makes it obvious which fields are static
Python and which are traced arrays, which is exactly what you need before
`vmap`-ing them ([4b](#4b-replace-per-endpoint-python-loops-with-vmap)).
Same for the endpoint dicts returned by `_endpoint_weights_and_r`, where
`'is_foundation'` is *absent* rather than `False` on coil-side entries and is
read with `ep.get('is_foundation', False)`.

### 3f. The non-linear Newton branch is ballast

`CuDSSNewtonSolver.newton_loop` has a full Newton iteration with residual
norms and host syncs (`cudss.py:457-485`) for a codebase whose only `Problem`
is `LinearElasticity3D`. The `linear=True` fast path is the correct one and is
16 lines. Given [A7](#a7--latent-trap-cudss-default-is-the-host-syncing-newton-loop),
consider deleting the iterative branch (and `tol`/`rel_tol`/`max_iter`) until a
non-linear problem exists; jax-fem's own `ad_wrapper` is available as the
fallback for anything non-linear.

---

## Rule 4 — JIT opportunities

### Measurements

**[measured]** CPU backend, 2 coils × 768 TET4 cells (288 nodes each),
12 beams, mean of 3 runs after warm-up:

| Block | Eager | Jitted | Speed-up |
|---|---:|---:|---:|
| `SupportBeams.geometry` + `.solve` | 195.5 ms | 0.89 ms | **219×** |
| `CoilFEM._body_force_at_quads` (one coil) | 27.8 ms | 0.79 ms | **35×** |
| `SupportBeams.compute_weights` (one coil) | 29.3 ms | 0.01 ms | **2000×** |
| `ElasticPipeline.solve` (umfpack) | 41.8 ms | — | not jittable |
| `recompute_fe_geometry` | 1.0 ms | — | |
| `surface_quad_points` | 1.0 ms | — | |
| `surface_jxw` | 1.5 ms | — | |

A full forward `CoilFEM.objective` with `coupling='staggered'` on this problem
takes **29.1 s**. The FEM solves account for 100 sweeps × 2 coils × 42 ms ≈
8.4 s. The remaining ~20 s is eager JAX dispatch in the beam assembly and the
weight functions. **On the CPU path, the un-jitted pure-JAX code costs more
than twice as much as the sparse solves it exists to feed.**

The one non-jittable step is `pipeline.fwd_pred`, because `jax_fem`'s
`ad_wrapper` runs scipy/PETSc on the host. Everything on either side of it is
pure JAX.

### 4a. JIT the beam side (biggest single win on CPU)

`SupportBeams.geometry`, `coo`, `solve`, `compute_attach`, `compute_weights`,
and `coupling_values` are pure JAX given static topology — `_endpoint_specs`
depends only on the static beam counts and `cc_groups`. The `SupportBeams`
instance can be captured by closure exactly the way `CoilFEM` is (philosophy
point 2 already says these should be pytree-like for this reason).

The payoff compounds: `solve` is called once per BG-S sweep, and the sweep
count is currently pinned at 100
([A2](#a2--block-gaussseidel-does-not-converge-and-says-nothing)). Jitting it
turns ~19.5 s of the 29 s evaluation into ~0.09 s.

The cleanest form is to jit one `_sweep(u_s, pts, bf, wt, sdofs, cdofs)`
function in `drivers.py` — the whole BG-S body minus `fwd_pred` — since
`geom`, `specs`, `surf_quad_pts`, and `jxw` are loop-invariant and would then
be hoisted by XLA rather than re-dispatched 100 times.

### 4b. Replace per-endpoint Python loops with `vmap`

`compute_weights`, `compute_attach`, `coupling_values`,
`_endpoint_weights_and_r`, and `_spring_stiffness_contributions` all iterate
`for spec in specs` in Python. That unrolls into `2·n_beams` copies of the
same subgraph. With 6 beams that is ~10 copies; a realistic 5-coil × 8-beam
network is ~90, and the trace grows linearly. Stack the endpoint data
(`x_ep`, `gamma3`, `sign_x`, `Q`, `coil`) into arrays grouped by coil and
`vmap` over the beam axis.

This matters even under jit, because trace-and-compile time is paid per shape
signature and the unrolled graph is what gets compiled. It also removes most
of [2h](#2h-endpoint-weights-and-moment-arms-recomputed-24-per-assembly)
mechanically.

`_spring_stiffness_contributions` additionally builds a Python list of 12×12
matrices and `jnp.stack`s them (`beam_network.py:1764-1812`), and
`_assemble_rhs` scatters into a `(n_support_dofs,)` array with a nested Python
loop and `.at[].add` per endpoint (`beam_network.py:1937-1972`). Both become
single vectorised scatters once the endpoints are stacked.

### 4c. JIT `_body_force_at_quads`

35× on CPU **[measured]**. Bind `coil_idx` with `functools.partial` (it selects
static mesh metadata). Everything inside — `interpax.interp1d`,
`B_self_quadrature`, `biot_savart`, `recompute_fe_geometry` — is jittable.

### 4d. JIT the metric block in `objective`

`recompute_fe_geometry` + `von_mises_on_quadrature` + the reduction is pure
JAX. Jit per coil with `metrics` as a static argument.

### 4e. Structure `objective` as three stages

```
_prepare(cdofs, idofs, sdofs) -> (pts, bf, wt, jxw, surf_pts, geom)   # jit
   per-coil fwd_pred                                                  # eager (host solver)
_metrics(pts, sols) -> totals                                         # jit
```

This is the shape the code already has; it needs two `jax.jit` boundaries and
a small amount of argument threading. It composes with the cuDSS path, where
the outer `jax.jit` in `CoilFEMObjective` subsumes both stages.

### 4f. `biot_savart` materialises an `(n_src, n_targets, 3)` intermediate

```175:181:src/coil_fem/magnetic.py
    def contrib_i(args):
        return field_from_one_source(*args)

    contribs = vmap(contrib_i)(
        (source_gammas, source_gammadashs, source_currents)
    )  # (n_src, n_targets, 3)
    return _BIOT_SAVART_PREFACTOR / n_quad * jnp.sum(contribs, axis=0)
```

For a W7-X-scale run (50 expanded coils, 10⁵–10⁶ FEM quadrature points) that
is 1–10 GB of intermediate before the reduction. The sum over sources is
associative, so a `lax.fori_loop` or `lax.scan` accumulation over sources
(or a chunked `vmap`) gives the same answer in `O(n_targets)` memory and is
jit-friendly at any mesh size. `field_from_one_source` also builds an
`(n_quad, 3)` `r` array per target inside a `vmap`; an `einsum` over
`(n_targets, n_quad, 3)` would let XLA fuse it.

### 4g. Halve the monolithic backward assembly

`make_merged_solve._fwd` calls `merged_solve`, which computes `V_merged`
internally and discards it; `_bwd` then calls `_assemble_merged_values` again
to rebuild the identical vector for the transposed solve
(`drivers.py:562-573`). Restructure `_fwd` to call `_assemble_merged_values`
once and stash `V_merged` in the residual tuple. This is one full assembly of
every coil Jacobian plus the beam blocks saved per gradient, on the priority
path.

Related: `_merged_constraint` recomputes `_surf_quad_pts` and `_surf_jxw`
(`drivers.py:576-577`) even though it already reuses `fwd_geom`. Combined with
`_assemble_merged_values` running in both `_fwd` and `_bwd`, the surface
geometry is built three times per gradient on top of the three times per
`set_params` noted in [2d](#2d-fe-geometry-recomputed-three-times-per-coil-per-evaluation).

### 4h. Two compilations where one would do

`CoilFEMObjective` builds and caches `jax.jit(value_and_grad(...))` and
`jax.jit(self._weighted_J)` separately (`objectives.py:170-176`). In a simsopt
loop `J()` is essentially always followed by `dJ()`, so the second compilation
mostly buys a cold-start cost. Consider dropping `_jit_J` and having
`_compute_J` delegate to `_compute_dJ` when a gradient is going to be needed
anyway, or keeping `_jit_J` only for the `run()`-style diagnostic path.

### 4i. Small hot-path items

- `CoilFEM.meshes` rebuilds a list on every access and is read inside
  `_solve_all`'s per-coil loop. Cache it.
- `_local_stiffness.single_beam` builds a 12×12 with 44 separate
  `.at[i, j].set(...)` calls (`beam_network.py:888-944`). Under `vmap` + jit
  XLA folds these, but a single `jnp.zeros((12,12)).at[rows, cols].set(vals)`
  with constant index arrays would be both faster to trace and easier to check
  against `docs/theory/bisymbeam.rst`.
- `SupportBeams.solve` densifies `K_ss` to `(n_dofs, n_dofs)` via
  `.at[I, J].add(V)` and calls `lineax.LU()`. The comment says
  "n_support_dofs is small in practice (e.g. 360)" — with a 5-coil × 8-beam
  network plus a wrap group it is ~1000+, and the dense LU is `O(n³)`. Since
  `K_ss` is block-diagonal by construction (`_build_static_ij` emits exactly
  one 12×12 block per beam), the "solve" is `n_beams` independent 12×12
  solves — `jnp.linalg.solve` on a `(n_beams, 12, 12)` batch, which is both
  exact and trivially vectorised. Worth checking whether the spring blocks
  break block-diagonality; from `_spring_stiffness_contributions` they do not
  (every contribution lands inside beam `b`'s own 12×12).

---

## Philosophy adherence

### Philosophy 1 — `CoilFEM` as a functional container

Largely honoured: static topology is fixed at construction, DOFs flow in
through `objective`/`run`, and the class is deliberately not a pytree.

The exception is `LinearElasticity3D.set_params`, which mutates
`self.shape_grads`, `self.JxW`, `self.v_grads_JxW`,
`self.physical_quad_points`, `self.nanson_scale`, `self.internal_vars`, and
`self.internal_vars_surfaces` on every forward pass. `objective` already works
around this by recomputing geometry externally (with a good comment
explaining why); `run` does not, and reads the mutated `shape_grads` via
`problem.von_mises_stress`. Each pipeline owns its own problem, so the value
happens to be right today — but it depends on which call last ran
`set_params`, which is a property of the driver, not of `run`. Pass
`shape_grads` explicitly, as `strain_tensors` already optionally allows.

### Philosophy 2 — `Support` must be independent of `CoilFEM`

Violated by the visualisation methods. `Support.plot_support(fem, ...)` and
`Support.save_support_vtu(fem, ...)` take the whole `CoilFEM` and reach into
five of its internals: `fem._compute_support_weights`, `fem._write_coil_vtu`,
`fem.pipelines[i].surface_node_indices`, `fem.meshes`, and
`fem.problem_options['winkler_k']`. Meanwhile `CoilFEM.plot_support` and
`CoilFEM.save_support_vtu` are pure forwards *to those methods*, so the
dependency is circular: `CoilFEM.save_support_vtu` → `Support.save_support_vtu(fem)`
→ `fem._write_coil_vtu`.

Suggested split, which also matches philosophy point 6:

- `CoilFEM` owns everything mesh-shaped (it has the meshes, the pipelines, and
  the writer) and keeps `plot_support` / `save_support_vtu`.
- `Support` exposes only geometry primitives it alone can produce — e.g.
  `SupportBeams.beam_segments(curves_jax, dofs) -> (N, 2, 3)` plus per-beam
  labels — and `CoilFEM` draws or writes them.

That removes ~200 lines of `fem.`-prefixed code from `supports.py` and
`beam_network.py` and makes `Support` genuinely standalone.

### Philosophy 3 / 4 — simsopt wrappers

`CoilFEMObjective` is clean: it reads DOFs live from the simsopt graph, caches
on `recompute_bell`, and wraps `CoilFEM.objective` / `run` without duplicating
logic.

`CoilSupportBeams.__init__` is ~300 lines that mix preset resolution, DOF
default generation, ragged-shape validation, fixed-mask construction, and
diagnostics. Three specific points:

- It `print()`s a six-line summary on construction (`optimizables.py:753-761`).
  A library that prints on every construction is awkward inside an
  optimisation loop or a test suite. Use `logging` or gate on a `verbose`
  flag, consistent with `CoilFEM.verbose`.
- `if kwargs is None:` (`optimizables.py:678`) is dead — `**kwargs` is always a
  dict.
- The nested helpers `_uniform_list`, `_zeros_list`, `_check_ragged_shape` are
  defined inside `__init__` and are ~70 lines of the total. `_uniform_list`'s
  stellsym branch in particular encodes real physics (why the last two groups
  use `[0, 0.5]` so reflections do not overlap) and deserves to be a
  module-level function with that reasoning in a docstring rather than an
  inline comment inside a closure.

### Philosophy 6 — logic should live where a dev would look for it

Mostly good — the clamp/attachment functions live next to the cross-sections
they belong to in `presets/cross_section_fns.py`, which is exactly the
intended locality.

Two misplacements:

- `CoilFEM.strain_energy_density` is a `@staticmethod` on `CoilFEM` whose only
  caller is `metrics.total_strain_energy`, which needs a lazy
  `from .coil_fem import CoilFEM` inside the function body to break the import
  cycle it creates (`metrics.py:283-290`). Move it into `metrics.py` and the
  cycle disappears along with the comment explaining it.
- The stellarator `Q` matrices belong in `geo/symmetries.py`
  (see [2b](#2b-rotation-and-symmetry-primitives-written-twice)).

### Cross-cutting: `magnetic.py` rewrites global JAX config at import time

```27:32:src/coil_fem/magnetic.py
# Importing simsopt pins JAX to the CPU backend (simsopt/geo/jit.py calls
# jax.config.update("jax_platform_name", "cpu")). Reset to the default so JAX
# keeps its normal backend auto-selection (GPU when available). This runs at
# import time, before any JAX computation, so it has no effect on an already
# initialised backend.
jax.config.update("jax_platform_name", None)
```

The intent is understandable, but it also overrides a *user's* explicit
choice. **[measured]** Setting `jax.config.update("jax_platform_name", "cpu")`
before `import coil_fem` is silently reverted, and the run then dies inside
`lineax` with `gpusolverDnCreate(&handle) failed: cuSolver internal error`.
`JAX_PLATFORMS=cpu` works because it is read at backend initialisation, but
that is not discoverable from the traceback.

Safer options: only reset when the current value is exactly `"cpu"` *and* it
was not set by the user (check `os.environ.get("JAX_PLATFORMS")` first), or
drop the reset and document the simsopt interaction in the README.

---

## Documentation drift

AGENTS.md describes the support architecture accurately in spirit but has
diverged in specifics:

| AGENTS.md says | Code says |
|---|---|
| `coupling/beam_networks.py` | `coupling/beam_network.py` |
| `SupportBeams(..., clamp_fn=...)` | `attachment_fn=...` |
| `compute_weights(coil_idx, surf_pts, curves_jax, dofs)` | `SupportBeams` adds a `geom=` kwarg |
| `coupling_terms(bcd, sdofs, surf_pts, coil_offsets, s_offset, surf_idx)` | Actual signature adds `surf_interp_by_coil` and keyword-only `jxw_by_coil`; and the drivers do not call it at all |
| `coo(bcd, sdofs, surf_pts)` | Adds `geom=` and keyword-only `jxw_by_coil` |
| `coupling='staggered'` (default) | `coupling='monolithic'` (`coil_fem.py:250`) |
| `Support` ABC with `solve` / `displacement_at` abstract | Plain class; nothing uses `abc`; `displacement_at` is unused |
| `clamp_fn` receives `(pts, dofs, sign_x)` | Receives `(pts, dofs, sign_x, beam_options)` |

`docs/developers/support_structure.rst` still walks a new implementer through
`displacement_at` as step 4, which is the dead method.

The working tree also carries 28 `.ipynb_checkpoints/` copies of package
modules under `src/coil_fem/` (~13,000 lines). They are correctly gitignored
and excluded from the wheel, but they are picked up by `rg`/`grep` and by
IDE and agent search, and several are stale enough to be actively misleading
(e.g. `src/coil_fem/.ipynb_checkpoints/container-checkpoint.py` is a 1,676-line
ancestor of `coil_fem.py`). Worth deleting from the working tree.

---

## Suggested sequencing

1. **Find the marginal mode.** [A2c](#a2c--the-support-stiffness-block-is-near-singular)
   is the one finding that affects the priority cuDSS path as much as the
   staggered one, and everything downstream of it (BG-S convergence, the
   accuracy of the dense `K_ss` LU, the conditioning of the merged matrix) is
   contingent on it. Check the two leads — CF foundation rotational grounding
   and the coil-side/foundation-side units mismatch — then add an assertion on
   `min(svd(K_ss))` (or surface the cuDSS inertia) so a zero-energy mode fails
   loudly at assembly.
2. **Unblock the rest of correctness.**
   [A1](#a1--jaxgrad-through-solve_staggered-raises) (decide: forward-only or
   `lax.while_loop` + real `custom_vjp`),
   [A2](#a2--block-gaussseidel-does-not-converge-and-says-nothing) (report
   non-convergence; revisit the `ω ≥ 0.1` clamp),
   [A4](#a4--three-of-the-four-cross-section-presets-are-unusable),
   [A5](#a5--cudss_mtype_id-has-two-conflicting-defaults),
   [A6](#a6--mutable-default-arguments-are-mutated-in-place). Add a regression
   test that takes `jax.grad` through a *real* `SupportBeams`, not a constant
   mock, and one that asserts BG-S actually converges rather than only that it
   returns.
3. **Delete.** Everything in [1a](#1a-dead-code-no-caller-in-src-tests-examples-or-docs),
   [1b](#1b-placeholder-implementations-that-the-architecture-no-longer-uses),
   [1d](#1d-single-use-wrappers-that-add-a-hop-without-adding-meaning), and the
   unused imports in [3d](#3d-unused-imports-and-locals-pyflakes). This shrinks
   the surface everything else has to be checked against.
4. **Unify.** One constitutive kernel
   ([2a](#2a-von-mises--cauchy-stress-written-three-times)), one Rodrigues,
   one set of symmetry transforms
   ([2b](#2b-rotation-and-symmetry-primitives-written-twice)), one
   `curves_from_dofs` ([2e](#2e-curves_jax-rebuilt-in-eight-places)), one FE
   and surface geometry cache
   ([2d](#2d-fe-geometry-recomputed-three-times-per-coil-per-evaluation)).
5. **Vectorise then JIT.** [4b](#4b-replace-per-endpoint-python-loops-with-vmap)
   first (it shrinks what has to be compiled), then
   [4a](#4a-jit-the-beam-side-biggest-single-win-on-cpu),
   [4c](#4c-jit-_body_force_at_quads), [4d](#4d-jit-the-metric-block-in-objective),
   [4e](#4e-structure-objective-as-three-stages).
6. **Priority-path polish.** [4g](#4g-halve-the-monolithic-backward-assembly)
   and [4f](#4f-biot_savart-materialises-an-n_src-n_targets-3-intermediate),
   which are the two items that matter most for large cuDSS/monolithic runs.
7. **Boundaries and docs.** [Philosophy 2](#philosophy-2--support-must-be-independent-of-coilfem),
   then reconcile AGENTS.md and `docs/developers/support_structure.rst`.

---

## Appendix A — measurement setup

All timings and error reproductions were produced in the `rod` conda
environment (`jax 0.6.2`, `jax_enable_x64=True`) with `JAX_PLATFORMS=cpu`,
using the fixtures from `tests/test_beam_networks.py`.

### Coil geometry used

This matters for [A2](#a2--block-gaussseidel-does-not-converge-and-says-nothing),
so it is spelled out. All coils are `CurveXYZFourierJAX` objects:

- **Coplanar circles** (`_make_curves` from `tests/test_beam_networks.py`):
  order-1, radius `1.0 + 0.1·i`, lying in the *xz*-plane with `y ≡ 0`, so the
  two coils are concentric and coplanar.
- **Offset non-coplanar circles**: the same, with the second coil translated to
  `y = 0.45` so the CC beam is neither radial nor in-plane.
- **Randomly perturbed coils**: order-2 and order-3, built by adding
  `amp · R · N(0, 1)` (`amp = 0.05`–`0.08`) to every sine and cosine
  coefficient of every coordinate above mode 1, plus an out-of-plane `y` mode-1
  term. These runs did not complete; see
  [A2d](#a2d--higher-order-curves-are-pathologically-slow).

Attachment functions: either `_uniform_clamp_fn` from the test fixture (weight
1.0 at every surface quadrature point) or a localised sigmoid ball,
`clamp_sigmoid(d_sq=‖x_beam_frame‖², r=0.06, sigmoid_eps=0.3)`, which clamps
9.4 % of the coil-0 surface area. Cross-sections came from
`_constant_section_fn()` (`A = 1e-4`, `Iy = Iz = 1e-8`, `J = 2e-8`), and
support DOFs from `_make_support_dofs` (CC attachment at `φ = 0.1` on both
ends, CF at `φ = 0.6`, foundation anchors offset `+0.5` in *x*).

### Configurations

Timing table: `n_base=2`, `n_beam_cc=3`, `n_beam_cf=3`, `nfp=2`,
`stellsym=False`, `N=32` quadpoints, mesh
`{'shape': 'rect', 'w1': 0.05, 'w2': 0.05, 'n_grid_1': 2, 'n_grid_2': 2}`
(288 nodes / 768 TET4 cells per coil), `winkler_k=1e8`, `solver='umfpack'`,
`coupling='staggered'`. Each entry is the mean of 3 calls after one warm-up
call, with `jax.block_until_ready` before and after timing.

Sweep count and gradient failure: the same, with `n_beam_cc=1`,
`n_beam_cf=1`, `N=8`, and a 1×1 cross-section grid, so the run completes in
~20 s.

Symmetry and conditioning study: the same, with `K_ss` assembled directly via
`support.coo(...)` and scattered into a dense array with `np.add.at`, and the
coupling blocks via `coupling_pattern` + `coupling_values`. `K_cc` was obtained
as `jax.jacfwd` of `problem.compute_residual_vars` at zero displacement, which
is the same tangent the solver factorises. No staggered solves were involved in
any of these.

Convergence study: `n_beam_cc=1`, `n_beam_cf=1`, `N=16`, mesh
`{'w1': 0.03, 'w2': 0.03, 'n_grid_1': 1, 'n_grid_2': 1}`, `k_tor=1e4` unless
noted, `k_lin ∈ {1e5, 1e8}`. Residuals were recovered by wrapping
`support.compute_attach` (to capture the incoming `u_s`) and `support.solve`
(to capture the outgoing `T(u_s)`); the effective `ω` was recovered as
`⟨u_{k+1} − u_k, Δ_k⟩ / ‖Δ_k‖²`. Runs with a capped sweep budget used
`functools.partial(solve_staggered, options={'max_iters': N})` monkeypatched
over `coil_fem.coil_fem.solve_staggered`, since `CoilFEM` does not expose
`options` ([1a](#1a-dead-code-no-caller-in-src-tests-examples-or-docs)).

### Reproduction sketches

The scratch scripts were not kept.

```python
# gradient failure
fem = CoilFEM(..., support=SupportBeams(...), coupling='staggered')
J = lambda cd, idd, sd: fem.objective(cd, idd, sd, metrics=('strain_energy',))['strain_energy']
J(cdofs, idofs, sdofs)                    # 0.36798387349276307
jax.grad(J, argnums=0)(cdofs, idofs, sdofs)  # ConcretizationTypeError

# sweep count
calls = [0]
orig = support.solve
support.solve = lambda inp: (calls.__setitem__(0, calls[0] + 1), orig(inp))[1]
fem.objective(cdofs, idofs, sdofs, metrics=('strain_energy',))
print(calls[0])                           # 100  (== max_iters)

# near-singular K_ss
geom = sb.geometry(curves_jax, sdofs)
I, J, V, n = sb.coo(curves_jax, sdofs, surf_quad_pts, geom=geom, jxw_by_coil=jxw)
K = np.zeros((n, n)); np.add.at(K, (np.asarray(I), np.asarray(J)), np.asarray(V))
s = np.linalg.svd(K, compute_uv=False)
print(n, s[0], s[-1], s[0] / s[-1])       # 48  1.543e+08  1.696e-02  9.097e+09
```
