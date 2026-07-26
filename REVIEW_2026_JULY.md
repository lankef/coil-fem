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

## Status

Reviewed 2026-07-24. **Revised 2026-07-25** with the author's decisions folded
in, so most items now read as *decisions* rather than options. Items marked
**Resolved** are already applied; **Excluded** means an unfinished feature that
is out of scope; **Declined** means considered and not taken.

## Executive summary

Two findings drive everything else:

1. **Staggered coupling is being retired.** `jax.grad` through it raises
   `ConcretizationTypeError`, the `custom_vjp` its docstring advertises as
   providing implicit-function-theorem gradients is bypassed even in
   principle, and the block Gauss–Seidel iteration never reaches a fixed
   point. `solve_staggered` will raise `NotImplementedError`; the numerical
   analysis is preserved in `notes/PLANS.md`. **[measured]**
2. **`k_tor ≠ k_lin` makes the coupled operator non-self-adjoint with an
   indefinite symmetric part, and this is *not* fixed by retiring staggered.**
   The same `K_ss` feeds the merged monolithic matrix, at `cond = 9.1 × 10⁹`
   with four negative eigenvalues out of 48. At `k_tor = k_lin` the matrix
   becomes symmetric to machine precision, positive definite, and four orders
   of magnitude better conditioned. **[measured]**

Behind those: roughly 500 lines have no caller anywhere in `src/`, `tests/`,
`examples/`, or `docs/`; the von Mises kernel is written out three times; the
beam-network assembly recomputes the same per-endpoint weights and moment arms
two to four times per evaluation; and on CPU the un-jitted pure-JAX code costs
more than twice as much as the sparse solves it feeds.

Priority ordering:

| Priority | Item | Where |
|---|---|---|
| P0 | `k_tor ≠ k_lin` leaves `K_ss` near-singular; inherited by the monolithic path | [A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular) |
| P0 | Retire the staggered driver and delete its body | [A1](#a1--staggered-coupling-is-disabled-by-decision), [2c](#2c-_sweep-and-_sweep_full-are-the-same-function) |
| P1 | Derive the cuDSS matrix type from per-block symmetry claims | [A5a](#a5a--fix-each-block-declares-its-own-symmetry-coilfem-takes-the-weakest) |
| P1 | Declare `is_linear` and delete the Newton branch; closes the jit trap | [3f-plan](#3f-plan--declare-linearity-on-the-problem-and-delete-the-loop), [A7](#a7--latent-trap-cudss-default-is-the-host-syncing-newton-loop) |
| P1 | JIT the beam side and the body-force block | [4a](#4a-jit-the-beam-side-biggest-single-win-on-cpu), [4c](#4c-jit-_body_force_at_quads) |
| P1 | Mutable default arguments mutated in place | [A6](#a6--mutable-default-arguments-are-mutated-in-place) |
| P2 | Dead-code removal | [Rule 1](#rule-1--does-this-need-to-exist) |
| P2 | Collapse the three von Mises implementations | [2a](#2a-von-mises--cauchy-stress-written-three-times) |
| P2 | Unify rotation and symmetry primitives into `geo/symmetries` | [2b](#2b-rotation-and-symmetry-primitives-written-twice) |
| P2 | Replace per-endpoint Python loops with `vmap` | [4b](#4b-replace-per-endpoint-python-loops-with-vmap) |
| P3 | Redundant geometry recomputation (volume 3×, surface 3×) | [2d](#2d-fe-geometry-recomputed-three-times-per-coil-per-evaluation) |
| P3 | `Support` reaches into `CoilFEM` privates for plotting/VTU | [Philosophy 2](#philosophy-2--support-must-be-independent-of-coilfem) |

**Resolved during the revision:** [A10](#a10--resolved-k_lin--k_tor-units-corrected)
(`k_lin` / `k_tor` documented as N/m³) and
[1e](#1e-resolved-copy-pasted-property-docstrings) (`nfp` / `stellsym` /
`beam_options` docstrings).

**Excluded as unfinished features:**
[A4](#a4--excluded-cross-section-presets-are-incomplete) (cross-section
presets) and [A9](#a9--excluded-disk-meshes-are-incomplete) (disk meshes).

See [Implementation phases](#implementation-phases) for the suggested order.

**Declined:** [1d](#1d-single-use-wrappers--declined) (single-use wrappers),
the `print()` summary and `kwargs is None` check in
[Philosophy 3 / 4](#philosophy-3--4--simsopt-wrappers), and the
`set_params` mutation in
[Philosophy 1](#philosophy-1--coilfem-as-a-functional-container), which is
idiomatic jax-fem.

---

## A. Correctness issues found while applying the rules

These are not rule violations as such, but they came out of the same reading
pass. Each carries the decision taken in the 2026-07-25 review pass.

### A1 — Staggered coupling is disabled by decision

**Decision: make `solve_staggered` raise `NotImplementedError`.** Coupled
supports use `coupling='monolithic'`. The full numerical analysis and the
measurements behind it are recorded in `notes/PLANS.md` under *Issue:
Staggered coupling is numerically unsound*; only the summary is kept here.

Three defects were found, of mixed kind, which is why the driver is being
retired rather than repaired piecemeal:

- **Numerical.** The block Gauss–Seidel map has an eigenvalue of essentially
  exactly 1 (measured `1.0000047`). The residual is flat from roughly sweep 7
  while `‖u_s‖` marches along a fixed direction with a per-sweep increment
  constant to four digits, so there is no fixed point to reach along that
  direction. Aitken cannot help, and the effective relaxation factor sits on
  its `max(0.1, ...)` clamp floor regardless. Independent of coil shape,
  attachment locality, and `k_lin`. **[measured]**
- **Numerical / modelling.** `k_tor ≠ k_lin` makes the coupled operator
  non-self-adjoint with an indefinite symmetric part. This one is **not**
  fixed by disabling the staggered driver — see
  [A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular).
- **Programming.** `jax.grad` through the driver raises
  `ConcretizationTypeError` from `float(jnp.max(jnp.abs(delta)))` in
  `_run_iterations`. The docstrings advertise `jax.lax.custom_root`, which does
  not exist anywhere in the file; the `_staggered_core` `custom_vjp` that does
  exist is an identity applied *after* the Python loop, is bypassed by
  `sol_list_by_coil` (the only output the objective consumes), and would
  double-count against the unrolled iteration history if it were reached.
  Separately, `_run_iterations` records `last_sol_list` *before* the Aitken
  update, so the returned `sol_list_by_coil` and `u_s` are one relaxation step
  out of sync. **[measured]**

The existing test does not catch the gradient failure because
`test_staggered_fixed_point_trivial` uses a mock support whose `solve` returns
a constant `jnp.zeros`, which keeps `delta` concrete.

**What to delete with it.** Once the function raises immediately, its whole
body is dead: `_sweep`, `_sweep_full`, `_run_iterations`, `_staggered_core`
with its `fwd`/`bwd` pair, and the unreachable `options` plumbing
([1a](#1a-dead-code-no-caller-in-src-tests-examples-or-docs)). See
[2c](#2c-_sweep-and-_sweep_full-are-the-same-function).

### A2 — Unequal `k_tor` and `k_lin` make the coupled operator near-singular

This is the finding that survives disabling the staggered driver, because the
same `K_ss` and coupling blocks are assembled into the merged monolithic
matrix.

Sweeping `k_tor` with `k_lin = 1e8` fixed **[measured]**:

| `k_tor` | `‖K_ss − K_ssᵀ‖/‖K_ss‖` | `‖K_cs − K_scᵀ‖/‖K_cs‖` | negative eigenvalues of `sym(K_ss)` | `σ_min(K_ss)` | cond `K_ss` |
|---:|---|---|---:|---|---:|
| 1e4 (fixture) | 1.249e-03 | 2.726e-02 | 4 / 48 | 1.696e-02 | 9.10e9 |
| 1e6 | 1.236e-03 | 2.699e-02 | 4 / 48 | 1.695e+00 | 9.10e7 |
| 1e8 (= `k_lin`) | **1.7e-17** | **0.0 exactly** | **0 / 48** | 1.611e+02 | 9.58e5 |

`σ_min(K_ss)` tracks `k_tor` linearly over four decades, and at
`k_tor = k_lin` the four negative eigenvalues of the symmetric part vanish, the
matrix becomes symmetric to machine precision, and the condition number drops
by four orders of magnitude. The weak subspace is exactly the rotational DOFs,
whose only external stiffness is `k_tor`.

The cause is structural rather than a tuning accident.
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

Two direct consequences of `cond = 9.1 × 10⁹` on a 48 × 48 matrix:

1. `SupportBeams.solve` densifies this block and factors it with
   `lineax.LU()` (`beam_network.py:2024-2037`), losing roughly 10 of 16
   digits.
2. The same values go into the merged monolithic matrix via
   `make_merged_solve`, so the conditioning is inherited by the cuDSS path.

Two further defects in the same rotational subspace, which compound it:

- **The CF foundation endpoint provides no rotational grounding at all.**
  `_endpoint_weights_and_r` sets `r_fnd = x_foundation[i][j] - geom['x_end'][b]`
  where `geom['x_end'][b]` *is* `x_foundation[i][j]` (the code says so at
  `beam_network.py:1704`), so `r_fnd ≡ 0`, hence `skew_sum = skew2_sum = 0`,
  hence `K_tr = K_rt = K_rr = 0` at the foundation node. `_assemble_rhs`
  explicitly does nothing for it (`pass`, `beam_network.py:1952`). Even setting
  `k_tor = k_lin` leaves the foundation nodes with no rotational spring,
  because `k_tor` multiplies a zero moment arm there.
- **A units mismatch between the coil side and the foundation side.** Every
  coil-side endpoint contributes `K_tt += k_lin · Σ(w · JxW) · I₃` — an
  *area*-weighted sum — while the foundation endpoint contributes
  `K_tt += k_lin · 1.0 · I₃` with a dimensionless `w_sum = 1.0`
  (`beam_network.py:1778-1781`). At `k_lin = 1e8` N/m³ the coil side is
  `≈ 7 × 10⁶` N/m and the foundation side is `10⁸` N/m — 14× stiffer, for no
  physical reason. The JxW change updated the coil branch and left the
  foundation branch alone.

A cheap diagnostic that would surface this at assembly time: `merged_solve`
already receives the matrix inertia from cuDSS and discards it
(`sol_flat, _inertia = solver_K(f_merged, csr_values)`, `drivers.py:559`).
Surfacing it, or asserting on `min(svd(K_ss))` in a test, would catch a
zero-energy mode where it happens.

**Open modelling question, worth settling before anything downstream:** should
`k_tor` remain a free parameter? A single foundation modulus
(`k_tor ≡ k_lin`, a genuine distributed spring bed) makes the merged system
symmetric positive definite for free. Both are N/m³ — see
[A10](#a10--resolved-k_lin--k_tor-units-corrected) — so a single modulus is
also the dimensionally natural choice. It would additionally collapse the
[A5a](#a5a--fix-each-block-declares-its-own-symmetry-coilfem-takes-the-weakest)
mechanism to a constant.

### A3 — Merged into A1

The out-of-sync `sol_list` / `u_s` return is a property of the staggered
driver's Aitken loop and is covered by
[A1](#a1--staggered-coupling-is-disabled-by-decision).

### A4 — Excluded: cross-section presets are incomplete

`solid_rectangle` and `solid_square` have no `*_attachment` function, so
`attachment_type='direct'` (the default) raises `ValueError` from `fetch_attr`
at construction; and `hollow_circle_attachment` is an alias of
`solid_circle_attachment`, which reads `dofs['r_beam']` — a key
`hollow_circle_dof_keys` does not provide.

**Excluded by review decision:** these are unfinished features rather than
defects. One adjacent point is worth keeping regardless: an `attachment_type`
that is neither `'direct'` nor `'wrap'` leaves `attachment_fn` unbound and
raises `UnboundLocalError`. Add an `else: raise ValueError(...)`.

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
  the sweep in [A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular). With
  the fixture's `k_tor = 1e4, k_lin = 1e8` the relative asymmetry is `1.2e-3`
  in `K_ss` and `2.7e-2` between `K_cs` and `K_scᵀ`, so `0` is required.

The hazard is therefore worse than a mistuned option. With `mview_id`
hard-coded to `0` (`CUDSS_MVIEW_FULL`, `coil_fem.py:441`) you hand cuDSS the
complete matrix and then tell it which triangle it may ignore; declaring
`SYMMETRIC` for the merged system yields the solution of a symmetrised matrix
with no error raised.

#### A5a — Fix: each block declares its own symmetry, `CoilFEM` takes the weakest

Rather than pick a default, let the matrix type be *derived* from claims made
by the objects that actually know the answer. No object infers another
object's properties, and no caller has to remember a cuDSS integer.

| Block | Owner of the claim | Claim | Why that owner |
|---|---|---|---|
| `K_cc` | `LinearElasticity3D` | `'symmetric'`, unconditionally | Property of the weak form. `k_tor` never enters it |
| `K_ss`, `K_cs` / `K_sc` | `Support` | base: `'symmetric'`; `SupportBeams`: `'symmetric'` if `k_tor == k_lin` else `'general'` | `k_tor` is the only asymmetry source anywhere in the system ([A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular)) |
| merged | `CoilFEM`, derived | weakest of all contributing claims | A merged matrix is only as symmetric as its least symmetric block |

**Vocabulary.** Solver-agnostic strings, matching the existing
`coupling='staggered'|'monolithic'` and `shape='rect'|'disk'` style. The
cuDSS-specific integers stay confined to `solvers/cudss.py`, which also
removes the `0=general, 1=symmetric, …` enum currently written out in two
separate docstrings (`coil_fem.py:131`, `solvers/__init__.py:49`) and keeps the
information reusable if the `'amgx'` entry in `_VALID_SOLVERS` ever grows a
backend.

```python
# solvers/cudss.py
_MTYPE_ID  = {'general': 0, 'symmetric': 1, 'spd': 3}   # cuDSS mtype_id
_STRENGTH  = {'general': 0, 'symmetric': 1, 'spd': 2}   # ordering for the meet

def weakest_symmetry(*claims: str) -> str:
    """Weakest (least-assuming) claim among ``claims``."""
    return min(claims, key=_STRENGTH.__getitem__)
```

```python
# problems/linear_elasticity.py
@property
def matrix_symmetry(self) -> str:
    """``'symmetric'`` — elasticity tangent, Winkler mass term, symmetric BC elimination."""
    return 'symmetric'

# coupling/supports.py — uniform contract; base support adds no asymmetry
@property
def matrix_symmetry(self) -> str:
    return 'symmetric'

# coupling/beam_network.py
@property
def matrix_symmetry(self) -> str:
    """``'symmetric'`` only when the torque and force laws share a modulus."""
    return 'symmetric' if self._k_tor == self._k_lin else 'general'
```

**Consumption points.** Two, and neither needs new plumbing:

- `build_fwd_pred(problem, problem_options)` already receives the problem, so it
  reads `problem.matrix_symmetry` itself. `CoilFEM` passes nothing.
- `build_monolithic_static` already holds `self.pipelines` and `self.support`,
  so it computes
  `weakest_symmetry(*[p.problem.matrix_symmetry for p in self.pipelines], self.support.matrix_symmetry)`
  locally.

**Override.** Keep `problem_options['cudss_mtype_id']` as an escape hatch, but
warn when it disagrees with the derived value, naming both. Silence is the
wrong default here because the failure mode is a wrong answer rather than an
exception. Split into `cudss_mtype_id_coil` / `cudss_mtype_id_merged` only if
overriding one path independently turns out to be needed.

**Constraints the implementation must respect:**

1. **Compare `k_tor == k_lin` exactly — do not reuse the `1e-6` relative
   tolerance** that `CoilFEM` applies to `winkler_k` vs `k_lin`
   (`coil_fem.py:313`). A `1e-6` relative difference in `k_tor` produces a
   `~1e-6` relative asymmetry, which is nowhere near machine zero, and with
   `mview = FULL` that silently symmetrises. **[measured]** the asymmetry is
   `1.7e-17` at exact equality and `1.2e-3` at `k_tor = k_lin·1e-4`. Both are
   Python floats set once from `beam_options`, so exact equality is both
   achievable and the only safe test.
2. **Never let anything auto-declare `'spd'`.** Definiteness is not a static
   property: **[measured]** `K_cc` is SPD with `w = 1` everywhere but has five
   eigenvalues within `1e-6` of zero with a 10 %-area clamp, because whether
   the six rigid-body modes are pinned depends on runtime weight values.
   Reserve `'spd'` for explicit opt-in. The cheap runtime verification already
   exists and is being thrown away: `merged_solve` receives the matrix inertia
   from cuDSS and discards it (`sol_flat, _inertia = solver_K(...)`,
   `drivers.py:559`).
3. **`mview_id` stays `0` (`CUDSS_MVIEW_FULL`)** — the full matrix is always
   supplied. Document the `mtype`/`mview` pairing next to `_MTYPE_ID` so the
   two cannot drift apart.

**Scope, and what actually changes.** The per-coil claim only alters behaviour
for `coupling='staggered'` with `solver='cudss'`: under `'monolithic'` the
per-coil `fwd_pred` is never called for a solve, and the uncoupled path only
ever sees a base `Support`, which already claims `'symmetric'`. So splitting
the ownership is about correctness of ownership and about not paying ~2× flops
and storage on the staggered path — not a broad speed-up. The one behaviour
change to flag is on the merged side: a user running `k_tor == k_lin` today
gets `mtype=0` (LU) and would get `mtype=1` (LDLᵀ) after this change, which is
both valid and cheaper, but is a change.

**Tests** (all matrix-assembly only; none require a solve):

- `SupportBeams.matrix_symmetry` returns `'symmetric'` at `k_tor == k_lin` and
  `'general'` otherwise, including at the boundary
  `k_tor = k_lin * (1 + 1e-12)`.
- `weakest_symmetry` returns `'general'` if any claim is `'general'`.
- Assemble `K_ss` plus the coupling blocks and assert
  `‖K − Kᵀ‖/‖K‖ < 1e-14` exactly when the derived claim is `'symmetric'`, and
  `> 1e-6` when it is `'general'`. This is the test that validates the claim
  against the matrix rather than against another line of code.
- `LinearElasticity3D.matrix_symmetry == 'symmetric'`, with the same assembled
  check on `K_cc` via `jax.jacfwd` of `compute_residual_vars`.
- An explicit `cudss_mtype_id` that disagrees with the derived value warns.

Finally, note that if `k_tor ≡ k_lin` were adopted as the model rather than
left as a free parameter (see
[A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular)), `SupportBeams`
would claim `'symmetric'` unconditionally and this whole mechanism would
reduce to a single constant — which is an argument for settling that modelling
question first.

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
breaks tracing in [A1](#a1--staggered-coupling-is-disabled-by-decision).
`cudss_linear` is not set anywhere in the repo, tests, or examples.

This only bites on `coupling='staggered'` with `solver='cudss'`, or the
uncoupled cuDSS path (the monolithic path never calls `fwd_pred`), but it is a
sharp edge.

**Fix: adopt the [3f plan](#3f-plan--declare-linearity-on-the-problem-and-delete-the-loop).**
`LinearElasticity3D` is affine in `u` by construction, so declaring
`is_linear = True` on the problem and deleting the iterative branch removes
both `float(jnp.linalg.norm(...))` calls outright. That closes this item rather
than working around it: with no host sync left in `newton_loop`, the `jax.jit`
that `CoilFEMObjective` switches on for `solver == 'cudss'` becomes valid
instead of latent, and the same construct can no longer break `jax.grad` the
way it does in [A1](#a1--staggered-coupling-is-disabled-by-decision).

Do not fix it the other way round — by leaving the loop and setting
`cudss_linear=True` at the call sites — since that leaves a default that is
wrong for every problem in the repo, and leaves the trap armed for the next
caller who omits the flag.

### A8 — Delete the dead default-from-support branch

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

**Decision: delete the inner branch (lines 306–310) now.** `winkler_k` stays
mandatory and must never be defaulted from `support.k_lin`. Delete it outright
rather than repairing it by hoisting the resolution above line 298 and relaxing
`_broadcast_problem_options`. Keep only the `else` arm — the equality
assertion. Reasons to record, since the dead branch reads as an intention
someone might otherwise restore:

- `winkler_k` is a property of the **coil-side discretisation**, not of the
  support. It is baked into `LinearElasticity3D` at construction through
  `additional_info` and is fixed for the life of the problem, so it must be
  known before the pipelines are built — which is why line 298 sits where it
  does.
- The key is required on the **uncoupled** path too, where no `k_lin` exists.
  Making it support-derived would leave the same option sometimes required and
  sometimes not, depending on the support class.
- Defaulting one modulus from the other hides a physical modelling choice. With
  the equality constraint in force, a user adjusting `k_lin` would silently
  change the coil FEM discretisation without ever naming the new value.
- It also makes `problem_options` a partly-output structure: line 309 rebuilds
  it with a value the caller never supplied.

#### A8a — The `hasattr` guard defeats the check it guards

Once the default-from-support branch is gone, the whole block exists only to
assert `winkler_k == support.k_lin`. That makes the guard actively harmful:

```305:305:src/coil_fem/coil_fem.py
        if self.support.is_coupled and hasattr(self.support, 'k_lin'):
```

`k_lin` is a **required** attribute whenever `is_coupled=True` (AGENTS.md
states this as part of the `Support` contract), so the `hasattr` tests for
something the contract already guarantees, and in practice it is always true
when reached: `SupportBeams.k_lin` is a property, and the base `Support` has no
`k_lin` but short-circuits on `is_coupled=False` first. Both halves of the
condition are effectively constant.

The problem is what happens if it is ever false. A coupled support genuinely
missing `k_lin` would **silently skip** the assertion rather than fail, leaving
the coil-side Winkler modulus free to disagree with the beam-side spring
modulus — precisely the inconsistency the block exists to prevent. Defensive
`hasattr` converts a contract violation into silent wrong physics.

Either declare `k_lin` on the base `Support` (as a property returning `None`
or raising `NotImplementedError`) and test unconditionally, or validate at
construction that `is_coupled=True` implies `k_lin` is present. Note also that
the tolerance here is `1e-6` relative, which is appropriate for a physical
consistency check but must **not** be reused for the exact `k_tor == k_lin`
symmetry test in
[A5a](#a5a--fix-each-block-declares-its-own-symmetry-coilfem-takes-the-weakest).

### A9 — Excluded: disk meshes are incomplete

`_broadcast_mesh_opts` accepts `shape='disk'` and `CoilMeshDisk` builds fine,
but `CoilMeshDisk` never overrides `_compute_uv_quad`, so `mesh.uv_quad` stays
`None` and `B_self_quadrature` raises `NotImplementedError` for disk sections
regardless. Any `CoilFEM` built with a disk cross-section therefore fails on
the first `run()` / `objective()`.

**Excluded by review decision:** an unfinished feature rather than a defect.

### A10 — Resolved: `k_lin` / `k_tor` units corrected

`SupportBeams` documented `k_lin` as `[N/m²]` and `k_tor` as `[N·m/m²]`, and
`CoilSupportBeams` repeated both. Dimensional analysis gives N/m³ for *both*:
`K_tt = k_lin · Σ(w·JxW) · I₃` with `Σ(w·JxW)` an area gives
`k_lin · m² = N/m`, and `τ = k_tor · Σ(w·JxW) · r × Δu` gives
`k_tor · m² · m · m = N·m`. Since `CoilFEM` also enforces
`winkler_k == support.k_lin` and documents `winkler_k` as N/m³, the two
docstrings were simply wrong.

**Resolved:** corrected in `beam_network.py:133-141` and
`optimizables.py:432-438`, and both now state the relationship to
`problem_options['winkler_k']`. That `k_lin` and `k_tor` share units is also
an argument for the single-modulus question raised in
[A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular).

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
| ~~`CurveXYZFourierJAX.kappa` / `.torsion` / `.frenet_frame`~~ | `curve_jax.py:144-216` | **Keep** — general-purpose curve geometry, retained deliberately |
| ~~`FramedCurveJAX.binorm` / `.torsion`~~ | `framed_curve_jax.py:548-554` | **Keep** — short aliases of `frame_binormal_curvature` / `frame_torsion`, retained deliberately |
| `problems.dirichlet_bc` | `linear_elasticity.py:83-123` | Exported in `problems.__all__`; the only references are its own docstring examples. `LinearElasticity3D` raises if `location_fns` is passed alongside `winkler_k_scalar` |
| ~~`HeatConduction3D`~~ | `problems/heat_conduction.py` | **Keep** — placeholder for planned thermoelastic work |
| `CoilFEM.n_total` | `coil_fem.py:274` | Assigned, never read |
| `solve_staggered(..., options=...)` | `drivers.py:123` | `CoilFEM._solve_all` never passes it, so `max_iters`, `atol`, `aitken`, `gmres_maxiter`, `gmres_tol` are all unreachable from the public API — directly relevant to [A2](#a1--staggered-coupling-is-disabled-by-decision) |
| `'amgx'` in `_VALID_SOLVERS` | `coil_fem.py:110` | Accepted by validation; `build_fwd_pred` would hand `{'amgx_solver': {}}` to jax-fem |
| `_HAS_SPINEAX` | `solvers/cudss.py:36` | Computed via `importlib.util.find_spec` and never read |
| Commented-out plot block | `coil_fem.py:1356-1364` | |

**Keep** `ThermoElasticPipeline` (`pipelines.py:282-295`) and
`HeatConduction3D`: both are deliberate placeholders for the planned
thermoelastic model. One small fix while they are here — `ThermoElasticPipeline.solve`
omits the `support_attach` parameter that `ElasticPipeline.solve` accepts, so on
the coupled path it raises `TypeError` before it can raise its intended
`NotImplementedError`. Match the signature so the stub fails with its own
message.

Everything else in the table above is to be removed.

### 1b. Placeholder implementations that the architecture no longer uses

**`displacement_at`.** `Support.displacement_at` returns zeros;
`SupportBeams.displacement_at` also returns zeros with a
`TODO(driver-integration)` note. Neither driver calls it — they use
`compute_attach`, which is the method that actually does the
rigid-body-displacement interpolation. AGENTS.md still lists
`displacement_at` as a required abstract method and
`docs/developers/support_structure.rst` devotes a step to implementing it.
**Decision: remove `displacement_at`** from both classes, and drop it from the
`Support` contract in AGENTS.md and from the walkthrough in
`docs/developers/support_structure.rst`. `compute_attach` is the method that
does the work.

**`coupling_terms`.** `Support.coupling_terms` (48 lines, `supports.py:527-578`)
and `SupportBeams.coupling_terms` (80 lines, `beam_network.py:1556-1635`)
have no caller in `src/`. `make_merged_solve` calls `coupling_pattern` and
`coupling_values` separately, which is the right split (static indices once at
construction, traced values every evaluation). `coupling_terms` just glues them
back together and is kept alive only by three tests. Note that
`SupportBeams.coupling_terms` also forgets to forward `geom`, so calling it
recomputes the beam geometry from scratch. **Decision: remove `coupling_terms`** from both classes and have the three
tests call `coupling_pattern` and `coupling_values` directly. Move its
physics-convention docstring — the clearest explanation of the `K_cs` / `K_sc`
sign conventions in the codebase — onto `coupling_values`, which is the method
that implements it.

### 1c. Base-class contracts that do not match the subclass

**Does a monolithic solve on a base `Support` reach `coo()`? No.** `_solve_all`
branches on `support.is_coupled` *before* any driver is selected, and base
`Support.is_coupled` is `False`, so the call goes to the inline per-coil loop.
`solve_monolithic` is never entered and `coo()` is never called. The guard in
`__init__` is the same: `build_monolithic_static` runs only when
`self.support.is_coupled and coupling == 'monolithic'`.

**And no, the base support should not force the staggered driver.** Each coil
being its own problem is precisely the *uncoupled* case, which already has its
own path — and forcing staggered would be strictly worse, since it would run
BG-S sweeps over a system that has no support DOFs to converge. The right home
for that path is the missing `solve_uncoupled` driver in
[2j-plan](#2j-plan--extract-solve_uncoupled-for-a-uniform-driver-contract);
with the staggered driver now raising
([A1](#a1--staggered-coupling-is-disabled-by-decision)), forcing it would in
fact break every uncoupled problem.

That said, the two base-class stubs are still wrong as contracts:

**`Support.coo(self)`** takes no arguments and raises `NotImplementedError`,
while `SupportBeams.coo(self, curves_jax, support_dofs,
surface_pts_by_coil=None, geom=None, *, jxw_by_coil)` takes five. The base
signature is unreachable today, but it is not a usable contract for a future
coupled support — anything calling it through the driver path would get
`TypeError` rather than the intended `NotImplementedError`. Give the base
method the real signature.

**`Support.solve`** returns `{}` while every caller does
`support.solve(inputs)['u_s']`, which is a `KeyError` rather than a graceful
no-op. Safe only because it is never reached when `is_coupled=False`. Return
`{'u_s': jnp.zeros(0)}` or raise.

### 1d. Single-use wrappers — declined

Excluded by review decision. The one item here that matters is
`CoilFEM.plot_support` / `save_support_vtu` forwarding into `Support` methods
that immediately reach back into `CoilFEM` privates; that is covered by
[Philosophy 2](#philosophy-2--support-must-be-independent-of-coilfem) and
should be addressed there instead. `utils.fetch_attr` is still covered by
[3a](#3a-stdlib-and-jax-built-ins).

### 1e. Resolved: copy-pasted property docstrings

Three `SupportBeams` properties carried the docstring of a fourth
(`"``True`` — beams have their own DOFs coupled to coil surface nodes."`), and
`nfp` and `beam_options` were annotated `-> bool`.

**Resolved:** `nfp` is now `"The number of field periods."` returning `int`,
`stellsym` is `"Stellarator symmetry."`, and `beam_options` is
`"Static beam configuration passed at construction."` returning `dict`
(`beam_network.py:392-405`).

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
  (`beam_network.py:283-294`).

**Decision: unify all of it into simple, readable *public* functions in
`geo/symmetries.py`**, imported by both the coil-expansion code and
`beam_network`. Concretely: promote `_rotate_points_z` and `_flip_points` to
public names, add the matrix form the beam network needs (e.g.
`stellsym_transform(tag, nfp) -> (3, 3)` covering `'none'`, `'flip'`,
`'flip_half'`, `'rotate'`), and put `rodrigues(axis, angle)` there so the two
existing copies collapse onto one. Beyond removing the duplication, this makes
it checkable in one place that the beam network's `'flip_half' = rot @ flip`
convention matches the coil expansion's `flip(rotate(x))` ordering — currently
verifiable only by reading two modules side by side.

### 2c. `_sweep` and `_sweep_full` are the same function

`drivers._sweep` (`drivers.py:230-266`) and `_sweep_full`
(`drivers.py:268-308`) differ only in whether `sol['sol_list']` is appended to
a second list — 35 duplicated lines including the `geom_kw` dance and the
`support_inputs` dict. `_sweep` exists only for `_staggered_core_bwd`.

**Decision: delete both**, along with the rest of the staggered body. Since
`solve_staggered` now raises `NotImplementedError` immediately
([A1](#a1--staggered-coupling-is-disabled-by-decision)), `_sweep`,
`_sweep_full`, `_run_iterations`, `_staggered_core` with its `fwd`/`bwd` pair,
and the `options` plumbing are all unreachable. What remains of the function is
the raise and its docstring.

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
`Support.save_support_vtu` (`supports.py:345-356`), and
`Support.plot_support` (`supports.py:238-249`). The two in `supports.py` are
identical for eleven lines, differing only in the loop header; the one in
`coil_fem.py` differs only in taking node positions from
`result['mesh_points'][i]` rather than recomputing them.

**Decision: add a helper in `supports.py`** that all three call, and **state in
its docstring that it exists solely for plotting and VTU export** — it is not
on the solve path and must not acquire callers there:

```python
def _support_weights_full(fem, coil_idx, pts_i, curves_jax, support_dofs):
    """Winkler weight per mesh node, zero off the Winkler surface.

    Plotting and VTU export only.  No solve path uses this; solvers consume
    per-surface-quadrature weights from ``Support.compute_weights`` directly.
    """
```

It takes `fem`, so position it to move together with the plotting code when
[Philosophy 2](#philosophy-2--support-must-be-independent-of-coilfem) is
addressed.

Note also that `save_support_vtu` and `save_run_vtu` then emit the *same* two
point fields — `support_weights` and `spring_k_Npm3 = weight_full * winkler_k`
— into two different files, so the identical quantity is computed and written
twice.

### 2g. `gamma3` resolution duplicated four times

```python
gamma3 = geom['gamma3'] if 'gamma3' in geom else self._direction_cosine_matrices(geom, dofs)
```

appears verbatim in `compute_weights` (`beam_network.py:1054`),
`compute_attach` (`:1330`), `coupling_values` (`:1506`), and `coo` (`:1887`).

**Decision: make `'gamma3'` a mandatory key of the `geom` dict.** `geometry()`
is the only public producer and always sets it, so the four fallbacks are dead
defensiveness that also silently masks a caller passing a bare
`_beam_geometry()` result. Replace each occurrence with `geom['gamma3']` and
let a missing key raise.

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
  **Decision:** keep the single definition on `CoilSupportFixed` and have
  `CoilSupportBeams` reuse it directly instead of redefining the body.
- `MonolithicStatic` stores `curve_qps` / `curve_orders`, duplicating
  `CoilFEM.base_curves_jax[i].quadpoints` / `.order`, and
  `surface_node_indices_by_coil`, duplicating `pipeline.surface_node_indices`.
- `CoilFEM.plot`'s boundary-face extraction (`coil_fem.py:1386-1403`) reruns
  the exterior-face detection that `LinearElasticity3D.custom_init` already
  did and stored in `boundary_inds_list` (`linear_elasticity.py:340-350`).
- `beam_network._check_beam_counts` and `coil_fem._broadcast_mesh_opts` both
  hand-roll "scalar or length-N sequence" broadcasting.

### 2j. `_solve_all` builds the same result dict twice; the uncoupled driver is missing

`_solve_all` ends in two hand-built dictionaries that agree on six of seven
keys:

```754:777:src/coil_fem/coil_fem.py
            return {
                'sol_list_by_coil': result['sol_list_by_coil'],
                'pts_by_coil':      pts_by_coil,
                ...
                'u_s':              result['u_s'],
            }

        # Uncoupled: independent per-coil solves
        sol_list_by_coil = [
            self.pipelines[i].solve(pts_by_coil[i], bf_by_coil[i], wt_by_coil[i])['sol_list']
            for i in range(n_base)
        ]
        return {
            'sol_list_by_coil': sol_list_by_coil,
            'pts_by_coil':      pts_by_coil,
            ...
            'u_s':              None,
        }
```

The only differences are where `sol_list_by_coil` comes from and whether `u_s`
is a value or `None`. The reason the second block exists inline rather than
behind a driver is simply that the uncoupled driver was never written — the
`drivers.py` module docstring already describes drivers as the things that
"replace the uncoupled per-coil loop inside `CoilFEM`", so the loop it refers
to is still sitting in `CoilFEM`.

#### 2j-plan — Extract `solve_uncoupled` for a uniform driver contract

Add the missing member so all three modes share one signature and one return
shape, and `_solve_all` reduces to a single dispatch and a single `return`:

```python
# coupling/drivers.py
def solve_uncoupled(pipelines, support, params) -> dict:
    """Independent per-coil FEM solves; no support DOFs.

    Returns the same keys as :func:`solve_staggered` with ``u_s = None``.
    """
```

Take `support` even though it is unused, so the three drivers are
interchangeable at the call site — that is what turns the dispatch into a
lookup rather than nested conditionals.

**One wrinkle to resolve first.** The contract is not actually uniform today:
`solve_monolithic(pipelines, support, params, static)` takes a fourth
argument. Two ways to fix it, either acceptable:

- bind it at construction with
  `functools.partial(solve_monolithic, static=self.monolithic_static)`, which
  also removes a `self.monolithic_static` read from the per-evaluation path; or
- move `static` into `params`, consistent with how every other driver input is
  already passed.

`_solve_all` then becomes roughly:

```python
driver = solve_uncoupled if not self.support.is_coupled else self._driver
result = driver(self.pipelines, self.support, driver_params)
return {
    'sol_list_by_coil': result['sol_list_by_coil'],
    'pts_by_coil':      pts_by_coil,
    ...
    'u_s':              result['u_s'],
}
```

**Also worth carrying through:** all three drivers return a `'diagnostics'`
key that `_solve_all` currently discards in both branches. A uniform contract
is where the non-convergence reporting from
[A2](#a1--staggered-coupling-is-disabled-by-decision) would land, so
plumb `diagnostics` out rather than dropping it.

**The payoff that matters most.** The uncoupled path is currently the only one
where `jax.grad` works on CPU — monolithic raises `NotImplementedError` unless
`solver='cudss'`, and staggered raises `ConcretizationTypeError`
([A1](#a1--staggered-coupling-is-disabled-by-decision)). Today that working path is
the fall-through `else` of a function whose coupled branch carries the dead IFT
`custom_vjp`, the host syncs, and the non-convergent loop. Separating them means
that acting on A1 — whether by rewriting `solve_staggered` around
`lax.while_loop` or deleting it and declaring the mode forward-only — cannot
regress the one path that works end to end.

**Explicit non-goals.** Keep the scope to the extraction. In particular, do
*not* also:

- add `'uncoupled'` to `_VALID_COUPLING`;
- derive the mode from `support.is_coupled` and validate an explicit
  `coupling` against it.

Dispatch stays keyed on `support.is_coupled` first, then `self.coupling`, as
it is now. (Noted for the record: this leaves
`CoilFEM(..., support=Support(), coupling='monolithic')` silently ignoring
`coupling`, and leaves `coupling='monolithic'` as a default that is untrue for
every base-`Support` problem. Deliberate — revisit separately if it ever
causes confusion.)

**Tests:** call each of the three drivers directly with the same
`driver_params` and assert the returned dict has identical keys; assert
`solve_uncoupled` reproduces the current per-coil results bit-for-bit on a
base-`Support` problem.

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
directly for the beam lines.

**Decision: call `meshio` directly at every write site** and drop the
intermediate wrapper, keeping `CoilMesh.meshio_cell_type` as the single
cell-type mapping. Note this does *not* subsume
[2f](#2f-surface-weight-scatter-duplicated-three-times): meshio has no concept
of point data on a subset of points, so the full-length weight array still has
to be built by the caller — and `plot_support`, one of the three 2f sites,
writes no file at all.

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
methods.

**Decision: use `typing.NamedTuple`.** It gives attribute access, typo
protection at the point of use, and — more usefully — makes it obvious which
fields are static Python and which are traced arrays, which is exactly what is
needed before `vmap`-ing them
([4b](#4b-replace-per-endpoint-python-loops-with-vmap)). Being a tuple, it is
also a JAX pytree, so a list of them flattens predictably if the endpoint data
is ever stacked.

Apply the same to the endpoint dicts returned by `_endpoint_weights_and_r`,
where `'is_foundation'` is *absent* rather than `False` on coil-side entries
and has to be read with `ep.get('is_foundation', False)` — a `NamedTuple` field
with a `False` default removes that asymmetry.

### 3f. The non-linear Newton branch is ballast

`CuDSSNewtonSolver.newton_loop` has a full Newton iteration with residual
norms and host syncs (`cudss.py:457-485`) for a codebase whose only `Problem`
is `LinearElasticity3D`. The `linear=True` fast path is the correct one and is
16 lines.

#### 3f-plan — Declare linearity on the problem and delete the loop

**Why it is safe.** `LinearElasticity3D`'s residual is exactly affine in `u`:
the stress `λ tr(ε − ε_th) I + 2μ(ε − ε_th)` has `ε_th` as a constant offset,
the body-force term does not involve `u`, the Winkler term
`∫ k w (u − u_att)·v dS` is affine, and Dirichlet conditions are imposed by
symmetric elimination. So `R(u) = K u − f`, and a single Newton step from
`u = 0` *is* the solution — the increment returned by `solve_step` is the
answer, with the post-step residual zero to round-off. Geometry enters through
`params['points']`, which is a parameter rather than the unknown, so nothing
in the differentiable pipeline makes the solve non-linear.

**Scope — smaller than it looks.** Two paths are unaffected:

- `coupling='monolithic'` already performs exactly one assemble and one solve.
  `solve_monolithic` goes through `merged_solve` and never calls
  `newton_loop`.
- The CPU backends are untouched. On `umfpack`/`petsc`, `build_fwd_pred`
  returns jax-fem's `ad_wrapper`, whose Newton loop lives inside
  `jax_fem.solver.solver()`; the `linear` flag never reaches it.

The change therefore affects `coupling='staggered'` with `solver='cudss'` and
the uncoupled cuDSS path — which is precisely where
[A7](#a7--latent-trap-cudss-default-is-the-host-syncing-newton-loop) bites.

**How.** Not by hard-coding `linear=True` inside `build_fwd_pred`.
`cudss_ad_wrapper` is a general utility, and one Newton step on a genuinely
non-linear problem is a *silently wrong answer* — a bad property to bake into
the wrapper. Declare it on the problem instead, as the same kind of static
claim as `matrix_symmetry` in
[A5a](#a5a--fix-each-block-declares-its-own-symmetry-coilfem-takes-the-weakest),
so the two live side by side and `build_fwd_pred` reads both from the object
it already receives:

```python
# problems/linear_elasticity.py
class LinearElasticity3D(DeviceProblem):
    is_linear = True                  # R(u) = K u - f exactly
    matrix_symmetry = 'symmetric'     # see A5a
```

Then **delete** the iterative branch rather than leaving it dormant, and have
`build_fwd_pred` raise on the cuDSS path when a problem declares
`is_linear = False`. That satisfies YAGNI — there is no non-linear problem in
the repo and `HeatConduction3D` is a stub
([1a](#1a-dead-code-no-caller-in-src-tests-examples-or-docs)) — while making a
future mistake loud instead of silent. jax-fem's `ad_wrapper` remains the
fallback for anything non-linear.

**What this removes:**

- the iteration branch, `cudss.py:457-485`;
- `tol`, `rel_tol`, `max_iter` on `CuDSSNewtonSolver` and `cudss_ad_wrapper`;
- the `cudss_tol`, `cudss_rel_tol`, `cudss_max_iter`, and `cudss_linear`
  option keys, plus their docstring entries in `coil_fem.py:133-135`,
  `solvers/__init__.py:50-55`, and `cudss.py:295-302`;
- the two `float()` host syncs, which is what closes
  [A7](#a7--latent-trap-cudss-default-is-the-host-syncing-newton-loop).

**One caveat: keep a way to detect a bad factorisation.** With no residual
check, a singular or near-singular `K` yields garbage silently — and that is
not hypothetical here, since
[A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular) measures
`cond(K_ss) = 9.1 × 10⁹`. The iterative branch at least reported a large
residual. The cheap replacement already exists and is being discarded: cuDSS
returns the matrix inertia at both `cudss.py:408`
(`inc, _inertia = self.cudss(...)`) and `drivers.py:559`, and both throw it
away. Surface it, or assert on it under a debug flag, and the deletion loses
nothing.

**Tests:** assert `‖K u − f‖ / ‖f‖ < 1e-10` after the single solve on a small
problem (which is the claim `is_linear` makes, checked against the matrix
rather than against another line of code), and assert that
`build_fwd_pred` raises for a stub problem with `is_linear = False`.

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
took **29.1 s**. The FEM solves account for 100 sweeps × 2 coils × 42 ms ≈
8.4 s. The remaining ~20 s is eager JAX dispatch in the beam assembly and the
weight functions. **On the CPU path, the un-jitted pure-JAX code costs more
than twice as much as the sparse solves it exists to feed.**

That end-to-end figure is measured through the staggered driver, which is
being retired ([A1](#a1--staggered-coupling-is-disabled-by-decision)). The
per-block speed-ups above are not staggered-specific: `SupportBeams.geometry`,
`.coo`, `compute_weights`, `coupling_values`, and
`CoilFEM._body_force_at_quads` are all on the monolithic path too. Retiring
staggered removes the 100× repetition, not the per-call eager overhead.

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
([A2](#a1--staggered-coupling-is-disabled-by-decision)). Jitting it
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
- `ElasticPipeline.__init__` calls `build_fwd_pred` unconditionally
  (`pipelines.py:75`), so on the monolithic cuDSS path every coil constructs a
  `CuDSSNewtonSolver` — CSR pattern, BC metadata, and a cuDSS device handle —
  that is then never used, since `solve_monolithic` goes through
  `merged_solve`. Build it lazily on first `solve()`.
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

### 4j. Higher-order curves are pathologically slow — unexplained

**[measured]** With everything else held fixed, swapping order-1 circles for
randomly perturbed order-2 or order-3 curves turns a case that runs in 3–8 s
into one that does not complete in over 10 minutes (three attempts, one at 20
minutes). The per-sweep work should not depend on the Fourier order — the
curve is evaluated once per objective call, not once per sweep — so the likely
culprits are distorted or inverted tetrahedra making the sparse solve
pathological, or repeated re-lowering of the RMF `lax.scan` that
`framed_curve_jax.py:226-238` already warns about.

This was found through the staggered driver, but nothing about it looks
staggered-specific, and randomly-shaped coils are the realistic optimisation
input. Worth reproducing on the monolithic path and profiling.

---

## Philosophy adherence

### Philosophy 1 — `CoilFEM` as a functional container

Largely honoured: static topology is fixed at construction, DOFs flow in
through `objective`/`run`, and the class is deliberately not a pytree.

`LinearElasticity3D.set_params` mutates `self.shape_grads`, `self.JxW`,
`self.v_grads_JxW`, `self.physical_quad_points`, `self.nanson_scale`, and
`self.internal_vars` on every forward pass, which looks like a departure — but
that is how jax-fem `Problem` subclasses are meant to work, and it is
**accepted as idiomatic**, not a finding.

The one thing worth carrying forward is a consequence rather than a cause:
`run()` reads the mutated `shape_grads` through `problem.von_mises_stress`
while `objective()` deliberately recomputes geometry externally, so the two
entry points use different code paths for the same quantity. That is resolved
by collapsing the duplicate von Mises implementations in
[2a](#2a-von-mises--cauchy-stress-written-three-times), not by changing
`set_params`.

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
default generation, ragged-shape validation, and fixed-mask construction.

**Decision: move `_uniform_list`, `_zeros_list`, and `_check_ragged_shape` to
module level** — about 70 lines of the total — and **comment each clearly as
being used only during initialisation**, so nobody mistakes them for runtime
helpers. `_uniform_list`'s stellsym branch in particular encodes real physics
(why the last two groups span `[0, 0.5]`, so that reflections neither overlap
nor intersect) which belongs in a docstring rather than an inline comment
inside a closure.

Two other observations here were **declined**: the `print()` summary on
construction (`optimizables.py:753-761`) is fine because simsopt objects are
constructed once per optimisation, and `if kwargs is None:`
(`optimizables.py:678`) is expected.

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

## Implementation phases

Each phase is independently mergeable and touches a largely disjoint set of
files.

### Phase 1 — Correctness of the coupled operator

Blocks Phase 3. Affects current monolithic results.

- Decide whether `k_tor` stays a free parameter, or is fixed to `k_lin`
  ([A2](#a2--unequal-k_tor-and-k_lin-make-the-coupled-operator-near-singular)).
- Give the CF foundation endpoint real rotational grounding (`r_fnd ≡ 0`).
- Fix the coil-side / foundation-side units mismatch (`Σ w·JxW` vs `w = 1.0`).
- Add an assembly-time guard: assert on `min(svd(K_ss))`, or surface the cuDSS
  inertia currently discarded at `cudss.py:408` and `drivers.py:559`.

### Phase 2 — Retire and delete

Pure subtraction. No behaviour change on the monolithic or uncoupled paths.

- Make `solve_staggered` raise `NotImplementedError` and delete its body
  ([A1](#a1--staggered-coupling-is-disabled-by-decision),
  [2c](#2c-_sweep-and-_sweep_full-are-the-same-function)).
- Delete the dead code in
  [1a](#1a-dead-code-no-caller-in-src-tests-examples-or-docs) and the
  placeholders in
  [1b](#1b-placeholder-implementations-that-the-architecture-no-longer-uses).
- Delete the dead default-from-support branch
  ([A8](#a8--delete-the-dead-default-from-support-branch)) and drop the
  `hasattr` guard ([A8a](#a8a--the-hasattr-guard-defeats-the-check-it-guards)).
- Fix the mutable default arguments
  ([A6](#a6--mutable-default-arguments-are-mutated-in-place)).
- Clear the unused imports and locals
  ([3d](#3d-unused-imports-and-locals-pyflakes)) and hoist the stdlib imports
  ([3b](#3b-local-imports-of-stdlib-and-core-dependencies)).
- Raise on an unrecognised `attachment_type`
  ([A4](#a4--excluded-cross-section-presets-are-incomplete)).

### Phase 3 — Solver claims

Depends on the Phase 1 `k_tor` decision.

- Declare `is_linear` on the problem and delete the Newton branch
  ([3f-plan](#3f-plan--declare-linearity-on-the-problem-and-delete-the-loop)),
  which closes
  [A7](#a7--latent-trap-cudss-default-is-the-host-syncing-newton-loop).
- Declare `matrix_symmetry` per block and derive the merged cuDSS type
  ([A5a](#a5a--fix-each-block-declares-its-own-symmetry-coilfem-takes-the-weakest)).

### Phase 4 — Unify duplicates

- One constitutive kernel
  ([2a](#2a-von-mises--cauchy-stress-written-three-times)).
- Public rotation and symmetry helpers in `geo/symmetries`
  ([2b](#2b-rotation-and-symmetry-primitives-written-twice)).
- One `curves_from_dofs` ([2e](#2e-curves_jax-rebuilt-in-eight-places)), one
  plotting-only weight helper
  ([2f](#2f-surface-weight-scatter-duplicated-three-times)), one `_clamp_fn`
  ([2i](#2i-smaller-duplications)).
- Mandatory `gamma3` key
  ([2g](#2g-gamma3-resolution-duplicated-four-times)).
- The missing `solve_uncoupled` driver
  ([2j-plan](#2j-plan--extract-solve_uncoupled-for-a-uniform-driver-contract)).
- Direct `meshio` calls
  ([3c](#3c-meshio-already-knows-how-to-write-these-files)) and the stdlib
  replacements in [3a](#3a-stdlib-and-jax-built-ins).
- Move the three init-only helpers to module level
  ([Philosophy 3 / 4](#philosophy-3--4--simsopt-wrappers)).

### Phase 5 — Vectorise and JIT

- `NamedTuple` endpoint specs
  ([3e](#3e-dataclasses-for-the-endpoint-spec-dicts)), then `vmap` the
  per-endpoint loops
  ([4b](#4b-replace-per-endpoint-python-loops-with-vmap),
  [2h](#2h-endpoint-weights-and-moment-arms-recomputed-24-per-assembly)).
- JIT the beam side
  ([4a](#4a-jit-the-beam-side-biggest-single-win-on-cpu)), the body force
  ([4c](#4c-jit-_body_force_at_quads)), and the metric block
  ([4d](#4d-jit-the-metric-block-in-objective)); then stage `objective`
  ([4e](#4e-structure-objective-as-three-stages)).
- Cache the FE and surface geometry
  ([2d](#2d-fe-geometry-recomputed-three-times-per-coil-per-evaluation)).
- Monolithic-path polish:
  [4g](#4g-halve-the-monolithic-backward-assembly),
  [4f](#4f-biot_savart-materialises-an-n_src-n_targets-3-intermediate),
  [4h](#4h-two-compilations-where-one-would-do),
  [4i](#4i-small-hot-path-items).

### Phase 6 — Boundaries and documentation

- Separate `Support` from `CoilFEM`
  ([Philosophy 2](#philosophy-2--support-must-be-independent-of-coilfem)).
- Move `strain_energy_density` into `metrics`
  ([Philosophy 6](#philosophy-6--logic-should-live-where-a-dev-would-look-for-it)).
- Stop overriding `jax_platform_name` at import
  ([Cross-cutting](#cross-cutting-magneticpy-rewrites-global-jax-config-at-import-time)).
- Reconcile AGENTS.md and `docs/developers/support_structure.rst`
  ([Documentation drift](#documentation-drift)).

### Deferred

- Investigate the higher-order-curve slowdown
  ([4j](#4j-higher-order-curves-are-pathologically-slow--unexplained)).
- The two open issues recorded in `notes/PLANS.md`.

---

## Appendix A — measurement setup

All timings and error reproductions were produced in the `rod` conda
environment (`jax 0.6.2`, `jax_enable_x64=True`) with `JAX_PLATFORMS=cpu`,
using the fixtures from `tests/test_beam_networks.py`.

### Coil geometry used

This matters for [A2](#a1--staggered-coupling-is-disabled-by-decision),
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
  [4j](#4j-higher-order-curves-are-pathologically-slow--unexplained).

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
