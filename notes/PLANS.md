# Plans for the project

## Todo
- Test staggered mode against monolithic mode in a simple, fixed-support case (low-res w7x)
- Add switch for monolithic vs staggered in CoilFEMObjective.  
- Integrate this test into the testing suite (monolithic vs staggered on cudss and scipy vs Dolfinx)


## Long term plans
1. Make meshio an optional req
2. Add field-on-coil to objective and investigate the best mechanism to calculate deformed dof and its jacobian wrt the initial dof.
3. Implement B_self without the uniform current assumption
4. Add the instrumentation to switch Lorentz off.
5. Add parameterized conductors and heat source/sinks for cross sections
6. Add beam-network model for cage-type support
7. Add volumetric grid for the topological optimization of shell/cage-type support. 
8. Add thermo-elastic model.

## Issues 

### Issue: Staggered coupling is numerically unsound (driver disabled)

Status: **Disabled**. `solve_staggered` raises `NotImplementedError`. Use
`coupling='monolithic'` for coupled supports. Recorded here so the analysis is
not lost; see `REVIEW_2026_JULY.md` A1 for the measurements.

Two separate numerical problems were found. The first is confined to the
staggered driver; the second is **not**, and still applies to the monolithic
path.

**1. The block Gauss-Seidel map has an eigenvalue of essentially exactly 1.**

On a 2-coil, 4-beam network the iteration never reaches a fixed point. The
residual `max|T(u) - u|` is flat from roughly sweep 7 onward while `||u_s||`
grows with a per-sweep increment that is constant to four digits. Successive
increments have `cos = 0.999999989` with a norm ratio of `1.0000047`, so the
iteration is translating along a fixed eigenvector with eigenvalue marginally
above 1 and there is no fixed point to reach along that direction. Aitken
cannot rescue it — no scalar relaxation produces a correction in a
unit-eigenvalue direction — and the effective factor sits on the `max(0.1, ...)`
clamp floor for most sweeps anyway.

The behaviour is independent of coil shape (concentric coplanar circles,
offset non-coplanar circles), of attachment locality (uniform whole-surface
clamp, 9.4 %-of-area sigmoid ball), and of `k_lin` (`1e5` and `1e8` give the
same relative behaviour, with the residual scaling as `1/k_lin`).

**2. `k_tor != k_lin` makes the coupled operator non-self-adjoint with an
indefinite symmetric part.**

With `k_lin = 1e8` fixed, sweeping `k_tor`:

| `k_tor` | asymmetry of `K_ss` | `K_cs` vs `K_sc^T` | neg. eigenvalues | cond `K_ss` |
|---:|---|---|---:|---:|
| 1e4 | 1.249e-03 | 2.726e-02 | 4 / 48 | 9.10e9 |
| 1e6 | 1.236e-03 | 2.699e-02 | 4 / 48 | 9.10e7 |
| 1e8 | 1.7e-17 | 0.0 exactly | 0 / 48 | 9.58e5 |

`sigma_min(K_ss)` tracks `k_tor` linearly, so the weak subspace is the
rotational DOFs, whose only external stiffness is `k_tor`. The cause is
structural: `_spring_stiffness_contributions` writes translation-rotation as
`-k_lin Σ w [r]x` and torque-translation as `+k_tor Σ w [r]x`, and since
`[r]x^T = -[r]x`, the two are transposes only when `k_tor == k_lin`. The same
holds for `K_cs` / `K_sc` in `coupling_values`. `k_tor` is therefore the only
source of asymmetry anywhere in the system.

**This one is not fixed by disabling the staggered driver.** The same `K_ss`
and coupling blocks are assembled into the merged monolithic matrix, and
`SupportBeams.solve` factors `K_ss` densely with `lineax.LU()` at
`cond = 9.1e9`, losing roughly 10 of 16 digits.

Two contributing defects in the same rotational subspace, worth fixing
regardless:

- The CF foundation endpoint has `r_fnd == 0` by construction
  (`geom['x_end'][b]` *is* `x_foundation[i][j]`), so `skew_sum` and
  `skew2_sum` are zero and the foundation node gets no rotational grounding
  at all — even at `k_tor == k_lin`, since `k_tor` there multiplies a zero
  moment arm.
- The coil-side branches now sum `w · JxW` (an area) while the foundation
  branch still sums a dimensionless `w = 1.0`, so at `k_lin = 1e8` the coil
  side is about `7e6` N/m and the foundation side is `1e8` N/m.

**Decided:** `k_tor` and `k_lin` are unified into a single
`beam_options['k_attachment']` [N/m³] — a genuine distributed spring bed, not
a workaround, since both were already the same units. This removes the
asymmetry/indefiniteness above entirely and unconditionally (not only when a
user happens to configure equal values): `K_ss` and the `K_cs`/`K_sc`
coupling blocks become symmetric to machine precision for every
configuration, and the condition number improves by four orders of magnitude
(`9.1e9 -> 9.58e5`). It also lets `SupportBeams.matrix_symmetry` inherit the
base `Support` claim unconditionally rather than compare `k_tor == k_lin` at
runtime — one fewer method on `SupportBeams`. See `REVIEW_2026_JULY.md` A2
and A5a.

**Resolved (Phase 1).** Both contributing defects above are fixed by the
CF-foundation hard-clamp introduced in `SupportBeams.coo()`:

- The CF foundation node-2 DOFs (rows and columns 6–11 of every CF beam's
  `12×12` stiffness block) are Dirichlet-eliminated with a unit diagonal,
  making the foundation a rigid anchor without any spring or moment-arm
  dependence. `r_fnd` and the area-integral inconsistency no longer appear.
- `K_ss` is now symmetric to machine precision for every configuration, and
  `min(svd(K_ss)) > 0` for any network with at least one CF beam.

The assembly-time guard requested above is also in place: the cuDSS inertia
that was silently discarded at `cudss.py` / `drivers.py` is now threaded
back into the diagnostics dict returned by `solve_monolithic`, and a new
`test_k_ss_symmetry` + `test_cf_beam_cantilever_analytic` test pair in
`tests/test_beam_networks.py` validates the fix.

### Issue: `winkler_k` may be ignored for CF-beam-only supports

Status: **Open, untested.** Noted while removing the dead
default-from-`k_lin` branch in `CoilFEM.__init__` (`REVIEW_2026_JULY.md` A6).

`winkler_k` remains a mandatory `problem_option` and is never defaulted from
`support.k_lin`. But when a `SupportBeams` has no fixed clamps
(`fixed_clamp_options={'enabled': False}`) and relies entirely on
coil-foundation beams, the coil-side Winkler weight field comes only from the
CF beam attachment functions — so `winkler_k` may end up scaling a weight
field that is zero or near-zero over most of the coil, and effectively be
ignored.

CF-beam accuracy is untested at the moment, so it is not yet clear whether
this is a real defect or expected behaviour. Needs a verification case before
any change is made.

### Issue: Newton Double Assembly in Linear Elasticity Solve

Status: **Deferred** (do not implement yet). Revisit only after the cuDSS
solver is confirmed to work robustly, to avoid introducing more moving parts
on top of an unverified solver.

Scope when implemented: **cuDSS path only**. The CPU jax_fem `solver()` double
assembly is library code and is intentionally.

Both the CPU path and the GPU (cuDSS) path solve the elasticity system with a
*general nonlinear* Newton loop, even though linear elasticity is linear.

jax_fem's CPU solver (`jax_fem/solver.py`):

```python
    res_vec, A = newton_update_helper(dofs)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = res_val
    rel_res_val = res_val/res_val_initial
    while (rel_res_val > rel_tol) and (res_val > tol):
        dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)
        res_vec, A = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val/res_val_initial
```

`newton_update_helper` calls `problem.newton_update(...)`, which assembles
**both** the residual vector **and** the full element tangent (stiffness)
matrix `A` (it fills `problem.V` / `V_jax`). Building `A` over a TET10 W7-X
mesh is the single most expensive step in the solve.

Coilforce's cuDSS path mirrors the same structure in
`src/coil-fem/cudss_solver.py` (`CuDSSNewtonSolver.newton_loop`, lines
462-489):

```python
        res_vec_bc = _get_res_and_update(dofs)
        res_val = float(jnp.linalg.norm(res_vec_bc))
        res_val_initial = res_val

        for _it in range(self.max_iter):
            rel_res = res_val / (res_val_initial + 1e-300)
            if res_val < self.tol or rel_res < self.rel_tol:
                break

            b = -res_vec_bc
            inc = self.solve_step(b, dofs)
            dofs = dofs + inc

            res_vec_bc = _get_res_and_update(dofs)
            res_val = float(jnp.linalg.norm(res_vec_bc))
```

where `_get_res_and_update` -> `problem.newton_update(sol_list)` re-assembles
the Jacobian (`V_jax`) on every call.

#### Why this assembles twice for a linear problem

Trace the loop for linear elasticity, where the stiffness `K` does not depend
on the displacement `u`:

1. **Assembly #1** at `u = 0`: gives `K` and residual `r(0) = K*0 - b = -b`
   (the load). `res_val` is large.
2. Enter the loop, solve `K*inc = -r(0) = b` -> `inc = K^-1 b = u*`, the
   *exact* solution. Set `dofs = u*`.
3. **Assembly #2** at `u = u*`: recomputes `K` (identical) and
   `r(u*) = K*u* - b ~= 0`. `res_val ~= 0` -> loop exits.

So the expensive tangent matrix `K` is assembled **twice**, but the second
assembly exists only to confirm the residual is zero. For a **direct** linear
solve (cuDSS, MUMPS, UMFPACK), `K*inc = b` is solved to machine precision, so
`r(u*) ~= 0` is *guaranteed*. The second assembly is pure wasted work - it
roughly doubles the assembly cost of every forward solve.

Note: the adjoint/backward pass already does only one assembly
(`adjoint_solve` / `implicit_vjp`), so this waste is forward-pass-only.

#### What is `K`?

`K` is the **global stiffness matrix** of the FEM system - the tangent
(Jacobian) of the residual with respect to the displacement DOFs. Linear
elasticity discretizes to a single linear system:

```
K u = b
```

- `u` - unknown nodal displacements (3 per node; length `n = num_total_dofs`).
- `b` - load vector (Lorentz body force + gravity integrated against basis
  functions, plus boundary terms).
- `K` - sparse `n x n` stiffness matrix; `K[i,j]` encodes how DOF `i`'s force
  response changes when DOF `j` is displaced:

```
K_ij = integral over Omega of  grad N_i : C : grad N_j  dV   + (Winkler surface terms)
```

In code, `K` is what `problem.newton_update(...)` builds - the per-element
Jacobian values stored flat in `problem.V` (CPU) or `problem.V_jax` (cuDSS),
which `assemble_csr_values` scatters into the CSR matrix cuDSS factorizes.

For a general nonlinear problem the Jacobian `A = dr/du` is re-evaluated each
iterate. For linear elasticity `r(u) = K u - b`, so `dr/du = K` is
**constant** - which is exactly why one assembly suffices.

### Caveat: direct vs iterative solvers

Single-assembly is exactly correct for **direct** solvers (cuDSS, MUMPS,
UMFPACK). For **iterative** solvers (`jax`/bicgstab, `amgx`), the post-solve
residual check is the only thing verifying convergence, so those should either
keep a single residual check (no Jacobian rebuild) or stay on the current
loop.

---

## Planned fix (cuDSS only; deferred)

#### Will this break nonlinear support?

Only if the loop is literally deleted. The plan keeps nonlinear capability:
add a single-assembly linear path and make it the default, while retaining
`newton_loop` for nonlinear use behind a flag.

#### Change 1: Add `linear_solve` to `CuDSSNewtonSolver`

In `src/coil-fem/cudss_solver.py`, add a method beside `newton_loop`:

- `set_params(params)`.
- Start `dofs = zeros(n)`.
- One assembly: `res_list = problem.newton_update(sol_list)` (fills
  `problem.V_jax`, gives `r(0) = -b`), then
  `res_vec_bc = apply_bc_vec(res_vec, dofs, problem)`.
- One solve: `inc = self.solve_step(-res_vec_bc, dofs)`; `dofs = dofs + inc`.
- Return `problem.unflatten_fn_sol_list(dofs)`.

`solve_step` already assembles `csr_values` from `V_jax`, applies symmetric
Dirichlet elimination, and calls cuDSS, so no other change is needed. This
performs exactly one Jacobian assembly.

#### Change 2: Route the wrapper through `linear_solve` (default), keep Newton

In `cudss_ad_wrapper`:

- Add parameter `linear: bool = True`.
- In `fwd_pred`, call `cudss_solver.linear_solve(params)` when `linear` is
  True, else `cudss_solver.newton_loop(params)`.
- Leave `newton_loop` and `adjoint_solve` unchanged, so nonlinear use remains
  available via `linear=False`.

#### Change 3: Docstrings

- Update the module docstring (item 5) and `cudss_ad_wrapper`'s docstring to
  note the default single-assembly linear forward solve and the `linear` flag.

#### No changes needed in coil_fem

`src/coil-fem/coil_fem.py` calls `cudss_ad_wrapper(...)` without `linear`, so
it defaults to the new single-assembly path automatically.

### Verification

- Confirm a cuDSS forward solve produces the same displacement / von Mises as
  before (single solve vs former loop) on a small case.
- A quick check that `linear_solve` output matches `newton_loop` output to
  solver tolerance is sufficient.

---

#### Why deferred

The cuDSS solver must be shown to work robustly first. Removing the double
assembly adds more moving parts to the forward path; doing it before the
baseline solver is trusted would make debugging harder. Implement only after
cuDSS is verified end-to-end.


