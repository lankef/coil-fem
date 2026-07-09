# Coupled-Support Boundary Condition: Plans A vs B

Design notes for the next-version support model: instead of Winkler springs
anchored to *fixed reference points*, couple the spring attachment points to the
**displaced coordinates of another linear system** (a second overlapping FEM
problem, or a linear beam-network problem), solved with the same sparse-direct
backend.

## Background: what exists today

The current Winkler support ties each surface quadrature point to its
*undisplaced* position. The surface term is `∫ k(x)·u·v dS` — **LHS-only**,
because the attachment target is implicitly `u_attach = 0`. It is implemented
with an identity surface map (`t = u`) and the stiffness `k(x)` folded into
`nanson_scale`, so **JAX-FEM assembly is untouched**
(`problem/linear_elasticity.py`: `get_surface_maps`, `set_params`).

Each coil is an **independent** `LinearElasticity3D` + its own
`ad_wrapper` (CPU) / `cudss_ad_wrapper` (GPU), solved as a single linear Newton
step with an adjoint `custom_vjp`.

## The generalization (common to both plans)

Coupling the spring to a moving target `u_B` gives the block system

```
[ K_A + C_AA    -C_AB      ] [u_A]   [f_A]
[ -C_BA         K_B + C_BB ] [u_B] = [f_B]
```

- `C_AA` = existing Winkler stiffness (already have it).
- `C_AB = C_BAᵀ` = transfer/interpolation operator sampling B's field at A's
  attachment locations (**new code, backend-independent**).
- `C_BB` = spring reflected onto B.

The **side-A change is small and needs no JAX-FEM patch**: JAX-FEM's surface
kernel already vmaps *surface internal variables* alongside `(u, x)`
(`jax_fem/problem.py::get_surface_kernel`, `compute_face`). So change the
surface map to `surface_map(u, x, u_attach) -> u - u_attach`, interpolate a new
`params['support_attach']` (n_surface_nodes, 3) to face-quad points using the
existing `_sel_face_sv`, and store it in `self.internal_vars_surfaces`. Because
`k(x)` stays in `nanson_scale`, this yields both the LHS stiffness *and* the RHS
load `∫ k·u_attach·v dS` for free.

The architectural weight is **not** the spring — it is introducing a second
solvable system + a transfer operator + a coupling/AD strategy. That choice is
Plan A vs Plan B.

---

## Plan A — Partitioned / staggered

Keep A and B as separate problems on the same backend and iterate:
1. solve B with current `u_A` as spring loads → `u_B`;
2. transfer `u_B` → A's `support_attach`, solve A → `u_A`;
3. repeat (with Aitken/relaxation) until the interface converges.

Sub-matrices are **constant across the interface sweeps** (only the RHS coupling
loads change), so the ideal cost is *factor each subproblem once, then cheap
solve-only sweeps*.

## Plan B — Monolithic

Assemble both problems + coupling into one matrix and factor once.

- On the **cuDSS path this is natural**: `cudss_ad_wrapper` already consumes raw
  COO (`problem.I/J/V_jax`) → one CSR + one factorization. Concatenate both
  problems' COO with a DOF offset, append coupling triplets, get one
  factorization + a trivially exact adjoint (A stays symmetric).
- On the **CPU `ad_wrapper` path this is invasive**: `jax_fem.solver.ad_wrapper`
  is bound to a single `Problem` and cannot express cross-mesh off-diagonal
  blocks; you would hand-write an assembler+solve, bypassing `jax_fem.solver`.

---

## 1. Performance comparison

| Dimension | A — Partitioned/staggered | B — Monolithic |
|---|---|---|
| Peak memory | **Best** — `max(factor A, factor B)` if sequential | Worst — factor of combined system ≥ sum |
| Largest solvable problem (VRAM-bound) | **Larger** — each factor fits one GPU | Limited by combined factor fitting one GPU |
| Multi-GPU | **Easy** — subproblems/coils independent | Needs distributed direct solve (cuDSS has MGMN; spineax likely single-GPU) |
| Factorizations per step (ideal) | 2 smaller + `k` cheap solve-only sweeps | 1 larger |
| Convergence robustness | Rate set by stiffness contrast; may need Aitken/relaxation; can be slow for strong two-way coupling | **Exact in one solve**, no iteration |
| Adjoint cost | Fixed-point adjoint reusing the same factors | Single adjoint solve reusing the factor (**cheapest**) |
| Coupling fill-in | None (subproblems separate) | Extra interface fill (modest — interface is a 2D manifold) |
| Shared support across many coils | **Scales** — solve shared B once, scatter, iterate | Couples *all* coils into one giant system — loses per-coil parallelism |

### Cost model (why the table looks that way)

Both backends use a **sparse direct solver**; in 3D the **factorization
dominates** (`~O(N²)` flops, `~O(N^{4/3})` fill/memory). Per optimization step
the geometry changes, so numeric factorization recurs every step regardless.

- Superlinearity means **two smaller factorizations (A) usually beat one bigger
  coupled factorization (B)**; the `k` interface sweeps are cheap *if* solve-only
  reuse is available.
- **Critical caveat:** the "cheap sweeps" advantage of A is **contingent on a
  factor-once/solve-many capability that spineax does not currently expose** (its
  warm path always `REFACTORIZATION + SOLVE`). Without it, A pays ~`k`
  factorizations/step, which erodes its per-step edge and can make B faster.
- B wins decisively on **robustness** (no interface iteration) and **adjoint
  simplicity**, at the cost of memory/scaling/multi-GPU.

### Rule of thumb

- One-way or weak coupling → A converges in ~1 sweep; strictly better.
- Strong two-way coupling, comparable stiffness → B's "exact in one solve" earns
  its keep, *if* the combined system fits one GPU.
- Shared support + many coils → strongly favors A.

---

## 2. Development steps

### Shared prerequisite (both plans)

1. **Nonzero attachment term (side A).** In `problem/linear_elasticity.py`:
   - `get_surface_maps`: `surface_map(u, x, u_attach) -> u - u_attach`.
   - `set_params`: interpolate `params['support_attach']` (n_surface_nodes, 3) to
     face-quad points via `_sel_face_sv`; store in `self.internal_vars_surfaces`.
   - `coil_fem.py::_forward_solve`: pass the new `support_attach` param.
   - *No JAX-FEM modification.* ~15–30 lines.
2. **Transfer operator `C_AB` / `C_BAᵀ`** (new module): interpolate B's field to
   A's attachment points and its transpose for the reaction. Must be
   differentiable if geometry is a DOF.
3. **Problem B** (new module): a second `LinearElasticity3D`/`DeviceProblem`, or
   a small linear beam-network problem, on the same backend.

### Plan A — additional steps

4. **Staggered driver**: block Gauss–Seidel over the interface with
   Aitken/dynamic relaxation; convergence check on interface displacement.
5. **Coupled AD**: wrap the fixed point in implicit differentiation
   (`jax.lax.custom_root` / a `custom_vjp` around the interface iteration). Each
   sub-solve keeps its own adjoint.
6. **(Performance, optional but important) solve-only reuse**: to realize the
   "factor once + cheap sweeps" cost, add a solve-only phase to **spineax**
   (`CUDSS_PHASE_SOLVE` reusing resident `state->data`, exposed as e.g.
   `solve_reuse(b)`), then call it in the sweep loop.
   - Requires forking/rebuilding the compiled CUDA extension (`--no-build-isolation`, `nvcc`, `cudss<0.8`).
   - **Correctness requirement:** each subproblem must own a **distinct
     persistent FFI state** so an alternating A/B loop never reads stale factors;
     verify XLA FFI `State` is scoped per solver instance, not per registered
     target.

### Plan B — additional steps

4. **Monolithic COO merge (cuDSS path)**: concatenate `problem.I/J/V_jax` of A
   and B with a DOF offset; append coupling triplets (`k·P`, etc.); build one CSR
   pattern (`build_csr_pattern`) and one `CuDSSNewtonSolver`.
5. **Single adjoint**: reuse the monolithic factorization (A symmetric) — mostly
   already handled by the existing `cudss_ad_wrapper` custom_vjp, extended to the
   merged system.
6. **CPU fallback decision**: either leave CPU forward-only/staggered, or write a
   custom assembler+solve bypassing `jax_fem.solver` (invasive). Monolithic is
   only "simple" on the cuDSS backend.

---

## Recommendation

Default to **Plan A (partitioned) with Aitken relaxation**, since it aligns with
the stated priorities (simplicity, minimal JAX-FEM change, large problems,
multi-GPU). Reach for **Plan B (monolithic-on-cuDSS)** only if measured interface
convergence is poor *and* the combined system fits a single GPU.

**Cross-cutting enabler:** a spineax **solve-only / factor-reuse** phase unlocks
both (a) Plan A's cheap interface sweeps and (b) removal of the existing
forward/adjoint double-factorization for the linear problem. It is the single
highest-leverage upstream feature for either path.
