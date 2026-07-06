# Issue: Newton Double Assembly in Linear Elasticity Solve

Status: **Deferred** (do not implement yet). Revisit only after the cuDSS
solver is confirmed to work robustly, to avoid introducing more moving parts
on top of an unverified solver.

Scope when implemented: **cuDSS path only**. The CPU jax_fem `solver()` double
assembly is library code and is intentionally left untouched.

---

## What "Newton double assembly" means

### The generic Newton loop

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

### Why this assembles twice for a linear problem

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

## What is `K`?

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

### Will this break nonlinear support?

Only if the loop is literally deleted. The plan keeps nonlinear capability:
add a single-assembly linear path and make it the default, while retaining
`newton_loop` for nonlinear use behind a flag.

### Change 1: Add `linear_solve` to `CuDSSNewtonSolver`

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

### Change 2: Route the wrapper through `linear_solve` (default), keep Newton

In `cudss_ad_wrapper`:

- Add parameter `linear: bool = True`.
- In `fwd_pred`, call `cudss_solver.linear_solve(params)` when `linear` is
  True, else `cudss_solver.newton_loop(params)`.
- Leave `newton_loop` and `adjoint_solve` unchanged, so nonlinear use remains
  available via `linear=False`.

### Change 3: Docstrings

- Update the module docstring (item 5) and `cudss_ad_wrapper`'s docstring to
  note the default single-assembly linear forward solve and the `linear` flag.

### No changes needed in coil_fem

`src/coil-fem/coil_fem.py` calls `cudss_ad_wrapper(...)` without `linear`, so
it defaults to the new single-assembly path automatically.

### Verification

- Confirm a cuDSS forward solve produces the same displacement / von Mises as
  before (single solve vs former loop) on a small case.
- A quick check that `linear_solve` output matches `newton_loop` output to
  solver tolerance is sufficient.

---

## Why deferred

The cuDSS solver must be shown to work robustly first. Removing the double
assembly adds more moving parts to the forward path; doing it before the
baseline solver is trusted would make debugging harder. Implement only after
cuDSS is verified end-to-end.
