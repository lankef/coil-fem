# Performance assessment: JIT, static caching, and the cuDSS path

Assessment of JAX performance inefficiencies across the codebase (2026-07-23).
Focus: missing JIT boundaries, integer/structural data that should be static,
jit-friendliness of the cuDSS interface, and caching of static results such as
sparse matrix patterns.

## TL;DR

The codebase is well set up for JIT (static topology, values-only tracing,
`custom_vjp` boundaries), but almost nothing on the hot path actually runs
under `jax.jit`. The biggest problems, in order of impact:

1. **A correctness-adjacent gradient cost bug in the staggered driver**: the
   sweep is re-linearized inside every GMRES matvec.
2. **`solve_monolithic` rebuilds all static structure (CSR pattern, cuDSS
   solver, coupling indices) on every call**, plus one wasted full Jacobian
   assembly per coil per call.
3. **The drivers create their closures (including `custom_vjp` functions)
   fresh on every evaluation**, which makes JIT caching impossible even if
   `jax.jit` were added — the fix is to hoist driver state into objects built
   once at `CoilFEM` construction.
4. **The beam network is assembled with per-beam Python loops and is
   recomputed from scratch ~5+ times per objective evaluation.**
5. **Nothing wraps the objective/gradient in `jax.jit`**, so every optimizer
   iteration pays full eager dispatch + retrace for Biot–Savart, self-field
   interpolation, mesh regeneration, and metrics.

**Key architectural fact** (verified in `spineax/cudss/solver.py`): spineax's
`CuDSSSolver.__call__` binds a custom JAX primitive backed by
`jax.ffi.ffi_call` (its `solve` wrapper is itself `jax.jit`-decorated with
`static_argnames=["device_id", "mtype_id", "mview_id"]`). The cuDSS solve is
therefore **fully traceable and jittable** — every blocker to a jitted
pipeline is in coil-fem code, not in the solver.

---

## 1. Staggered adjoint: `jax.vjp(_sweep, ...)` inside the GMRES matvec

`src/coil_fem/coupling/drivers.py`, `_staggered_core_bwd`:

```python
def A_T_fn(v):
    _, vjp_fn = jax.vjp(_sweep, u_s_star)   # <-- inside the matvec!
    sweep_vjp = vjp_fn(v)[0]
    return v - sweep_vjp
```

`jax.vjp` runs the **full forward sweep** — one FEM solve per coil plus the
support solve — to build the linearization. Because it sits inside `A_T_fn`,
this happens on *every GMRES iteration* (up to `gmres_maxiter=200`). Hoisting
one line out of the closure:

```python
_, vjp_fn = jax.vjp(_sweep, u_s_star)   # linearize ONCE
def A_T_fn(v):
    return v - vjp_fn(v)[0]
```

turns O(gmres_iters) forward FEM solves into exactly one. This is likely the
single largest win for gradient evaluations on the staggered path.

## 2. `solve_monolithic`: static structure rebuilt every call

`solve_monolithic` is invoked from `CoilFEM._solve_all` on **every**
`objective()` / `run()` call, and everything in it is per-call:

- **CSR pattern**: `build_csr_pattern(I_merged, J_merged, n)` does a host-side
  lexsort + unique over the full merged nnz (millions of entries for realistic
  meshes) — per call. The docstring says "built once on the host … and reused
  by every forward and backward evaluation", but the reuse only holds *within*
  one call; the cache dies when the function returns. Same for the lazily
  built adjoint pattern (`_get_adjoint`).
- **`CuDSSSolver` objects** are re-instantiated per call (forward and
  adjoint).
- **A wasted full Jacobian assembly per coil**: the pattern-probing loop calls
  `pipeline.assemble_coo(...)`, which triggers `set_params` +
  `compute_newton_vars` — a complete device assembly — and then throws away
  `V` and `load`, keeping only `I`/`J`. But `problem.I` / `problem.J` are
  static attributes available since construction; no assembly is needed to
  read them. `_assemble_merged_values` then assembles again for the actual
  solve. Net: 2× assembly per forward, 3× per gradient (the backward
  re-assembles too).
- **Coupling indices are traced when they should be static.** In
  `SupportBeams.coupling_terms`, `I_cs`/`J_cs`/`I_sc`/`J_sc` are built from
  purely static data (`surf_idx`, DOF offsets, `d3 = jnp.arange(3)`) but with
  `jnp` ops, producing device arrays. `solve_monolithic` then does
  `onp.asarray(coupling0['I_cs'])` — a device→host transfer per call — and the
  backward `_merged_constraint` recomputes the same indices as traced values,
  forcing dynamic scatters. Build them once with numpy at construction (like
  `SupportBeams._build_static_ij` already does for `K_ss`) and return only
  `V_cs`/`V_sc` as traced.

**Fix shape**: build a persistent monolithic-driver object (at
`CoilFEM.__init__`, or lazily on first solve) holding: merged I/J, forward +
transpose CSR patterns, both `CuDSSSolver` instances, the static coupling
index arrays, and the `custom_vjp`-wrapped solve function. Per evaluation only
COO *values* and the RHS get recomputed. Once the closure is created a single
time, wrapping the value-assembly + scatter + FFI solve in `jax.jit` is
straightforward, because:

- spineax's `solve` is already a jitted primitive binding,
- `assemble_csr_values` / `apply_symmetric_dirichlet` are already jitted,
- the `gpu_assembly=True` assembly path in `DeviceProblem.compute_newton_vars`
  is pure `jnp` and traceable.

The reason JIT wouldn't help *today* is that `_merged_solve` and its
`custom_vjp` pair are re-defined inside `solve_monolithic` on every call — a
fresh function identity means a fresh trace cache every time. Hoisting the
closure is a prerequisite for jitting, not just a cleanup.

## 3. `CuDSSNewtonSolver.newton_loop` (uncoupled cuDSS path)

`src/coil_fem/solvers/cudss.py`:

- `float(jnp.linalg.norm(...))` forces a host sync per Newton iteration, and
  `_get_res_and_update` (residual + Jacobian assembly) runs eagerly —
  thousands of small dispatches per call. `_get_res_and_update` plus
  `solve_step` are pure-JAX (on the `gpu_assembly=True` path) and could be one
  jitted function.
- The problem is **linear elasticity**: K doesn't depend on u. One assembly +
  one solve is exact; the current loop does an initial residual, one solve,
  and a second residual assembly just to confirm convergence — 2 assemblies
  where 1 suffices. A `linear=True` flag that does assemble-solve-return would
  remove a third of the per-solve cost and all host syncs.
- Minor: with Winkler-only supports there are no Dirichlet DOFs, so
  `bc_dof_mask` is all-False and `apply_symmetric_dirichlet` is a full-nnz
  no-op each solve. Cheap under jit; skippable statically since
  `bc_dof_mask.any()` is known at construction.

## 4. `SupportBeams`: per-beam Python loops, recomputed many times per evaluation

Two independent issues in `src/coil_fem/coupling/beam_network.py`:

**Redundant recomputation.** One monolithic objective evaluation recomputes
`_beam_geometry` + `_direction_cosine_matrices` + `_endpoint_specs` + all
`attachment_fn` calls:

- once per coil inside `compute_weights` (via
  `CoilFEM._compute_support_weights`, so `n_base` times),
- again in `support.coo` (inside `_assemble_merged_values`),
- again in `coupling_terms`,
- again in `_endpoint_weights_and_r` (called from `coo`),
- and all of the above again in the backward pass.

That's roughly `n_base + 3` full beam-geometry evaluations per forward and
double per gradient. The geometry depends only on
`(base_curves_dofs, support_dofs)`, so it should be computed once per
evaluation and threaded through (or, at minimum, live under one jitted
function where the recomputation is at least compiled — computing once is
still better).

**Eager per-beam scalar loops.** `_beam_geometry` calls
`curve.gamma_eval(phi_s)` per beam per endpoint with a *scalar* phi — each
call builds the full Fourier basis (`arange`, `sin`, `cos`, stack) as a
separate eager dispatch chain. `gamma_eval` already accepts array `phi`;
batching per (curve, group) collapses dozens of dispatch chains into one.
Same pattern in `_spring_stiffness_contributions` (Python loop building 12×12
blocks with `.at[].set`), `coupling_terms` (loop over 2×n_beams specs), and
`compute_attach`. Beam counts are static tuples, so these all vectorize
cleanly with `vmap` + segment-sum-style reductions, or at minimum become cheap
once compiled under a single jit.

## 5. No JIT at the top level

`CoilFEM.objective` and `CoilFEMObjective._weighted_J` / `value_and_grad` run
fully eagerly. Consequences per optimizer iteration:

- **Biot–Savart** (`biot_savart`, the largest dense op: `n_base` separate
  calls, each over all coils × all quad points) executes unfused with large
  materialized intermediates.
- `B_self_quadrature` runs six `interpax.interp1d` cubic-spline fits per coil,
  each an eager tridiagonal solve.
- `recompute_fe_geometry`, metrics, symmetry expansion — all re-dispatched op
  by op every call, and `value_and_grad` re-traces the whole graph every call.

On the **CPU path** (`ad_wrapper` → umfpack/scipy host solves) end-to-end jit
is impossible, but the pre/post-processing (body force, weights, metrics)
could still be jitted as separate chunks. On the **cuDSS path** there is no
fundamental blocker: after the driver restructure in §2,
`jax.jit(jax.value_and_grad(weighted_J, argnums=(0, 1, 2)))` created once at
construction should work — `support_dofs` dicts of arrays are valid pytree
inputs, `metrics` is already static, and mesh topology fixes all shapes. One
caveat to watch: `set_params` mutating `problem.internal_vars` with traced
arrays works only when write and read happen inside the same trace — which
they do in the current call structure, but it's the fragile part of jitting
and worth a test.

Related duplications at this level:

- `CoilFEM.run` calls `_body_force_at_quads` a second time for every coil to
  get `B_self`/`B_ext`, after `_solve_all` already computed the identical body
  force — the full Biot–Savart cost twice per diagnostic run. Returning the
  B-fields from the first pass fixes it.
- `CoilFEMObjective`: an optimizer calling `J()` then `dJ()` runs the forward
  solve twice — `_compute_J` evaluates eagerly, then `_compute_dJ`
  re-evaluates inside `value_and_grad`. Since `dJ` refreshes the J cache
  anyway, `_compute_J` could simply delegate to `_compute_dJ` (or a shared
  `value_and_grad` with the gradient computed lazily).

## 6. Smaller / static-integer observations

- `drivers.py` staggered `_run_iterations`: Python loop with `float()` syncs
  per BG-S iteration is documented and acceptable eagerly, but the final
  "recovery" sweep re-solves every coil one more time after convergence;
  caching the last sweep's `sol_list` when the residual passes tolerance saves
  one full multi-coil solve per evaluation. Long-term, a `lax.while_loop` body
  would make the whole driver jittable.
- `beam_network.py` properties `nfp`, `beam_options`, `stellsym` have
  copy-pasted wrong docstrings ("``True`` — beams have their own DOFs…").
- `coil_fem.py` `n_nodes`/`n_cells` properties are typed `int` but return
  lists.
- `_body_force_at_quads`: `jnp.broadcast_to((I / A) * t_hat_q[:, :, :], ...)`
  is a no-op wrapper; the interpax import inside the function is per-call
  (trivial but pointless).
- Static integers are mostly handled well already (`n_beam_cc` tuples,
  `phi_quad`, `surface_node_indices`, `nnz_csr` static in the jitted scatter).
  The only real "should-be-static" offenders are the coupling-term index
  arrays (§2) and, marginally, `surface_node_indices` being stored as `jnp`
  int32 where numpy would avoid host↔device round-trips during pattern
  building.

---

## Implementation status

### Completed

| Item | Location | Status |
|------|----------|--------|
| Hoist `jax.vjp` out of GMRES matvec (§1) | `drivers.py _staggered_core_bwd` | **Done** |
| Split `coupling_terms` → static `coupling_pattern` + traced `coupling_values` (§2 partial) | `beam_network.py`, `supports.py` | **Done** |
| Remove `base_curves_jax` from `SupportBeams`; pass `curves_jax` to methods explicitly | `beam_network.py`, `drivers.py`, `coil_fem.py` | **Done** |
| `MonolithicStatic` frozen dataclass + `CoilFEM.build_monolithic_static(solver)` (§2, §3) | `drivers.py`, `coil_fem.py` | **Done** — pattern built once at construction; no probe `assemble_coo`; coupling I/J from pure-numpy `coupling_pattern`; forward + adjoint CSR + cuDSS handles built once |
| `SupportBeams.geometry()` — compute `_beam_geometry` + `_direction_cosine_matrices` once (§4 partial) | `beam_network.py` | **Done** — `compute_weights`, `compute_attach`, `coo`, `coupling_values`, `solve` all accept `geom=None`; drivers and `CoilFEM._solve_all` compute geometry once and thread it through |
| `Support.geometry` base-class stub — removes 4 `hasattr` guards | `supports.py`, `drivers.py`, `coil_fem.py` | **Done** |
| `_compute_support_weights` accepts `curves_jax` directly — eliminates O(n_base²) redundant curve constructions per forward | `coil_fem.py` | **Done** |
| Drop `'base_curves_dofs'` dead key from `params` dict | `coil_fem.py`, `drivers.py` | **Done** |
| Inline `_forward_solve` one-liner shim | `coil_fem.py` | **Done** |
| Fix `n_nodes`/`n_cells` type annotations (`-> int` → `-> list[int]`) | `coil_fem.py` | **Done** |
| Save `geom` in `_fwd` residuals for `make_merged_solve` — avoids one geometry recompute per backward | `drivers.py` | **Done** |
| Cache `sol_list` at BG-S convergence — eliminates redundant recovery sweep | `drivers.py` | **Done** |
| Deduplicate Biot–Savart in `run()` — `_solve_all` now returns `B_self_by_coil`/`B_ext_by_coil` | `coil_fem.py` | **Done** |

### Deferred follow-ups

**`CoilFEMObjective` J/dJ ordering:** When `J()` is called before `dJ()`, two forward passes occur. The current caching policy is intentional — `J()` is kept as a cheap forward-only call for line-search use cases. Calling `dJ()` first is always efficient since `_compute_dJ` refreshes the J cache as a side-effect of `value_and_grad`.

### Recently completed

| Item | Location | Status |
|------|----------|--------|
| Batch `gamma_eval` calls in `_beam_geometry` — one call per (curve, group) instead of one per beam; `jnp.concatenate` final assembly | `beam_network.py` | **Done** |
| Batch theta-flattening in `_direction_cosine_matrices` — one `concatenate` per group instead of one append per beam | `beam_network.py` | **Done** |
| `CuDSSNewtonSolver` `linear=True` fast path — single assemble + solve, no iteration loop, no host syncs | `solvers/cudss.py`, `solvers/__init__.py` | **Done** — expose via `problem_options['cudss_linear'] = True` |
| Cache `jax.jit(value_and_grad(weighted_J))` at construction in `CoilFEMObjective` — cuDSS path only; CPU/staggered path unchanged | `simsopt/objectives.py` | **Done** — `_jit_vg` / `_jit_J` compiled once, reused on each optimizer step |
