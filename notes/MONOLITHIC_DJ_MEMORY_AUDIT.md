# Monolithic + SupportBeams: `dJ()` memory audit

Audit date: 2026-08-03. Focus: why `CoilFEMObjective.dJ()` peaks substantially higher than a forward objective evaluation under `coupling='monolithic'` + `SupportBeams` + `solver='cudss'`.

Scope: call graph, concrete memory amplifiers, ranked tips. No code changes in this note.

---

## Verdict

Monolithic forward already pays for one large GPU factor of the merged system. **`dJ()` adds four more costs on top:**

1. An outer reverse-mode tape through FE geometry, Biot–Savart / Lorentz body forces, and metrics.
2. Retention of the full merged COO `V_merged` (plus solve inputs) in the `custom_vjp` residual.
3. A **second** cuDSS factor workspace (`solver_KT`) for the adjoint \(K^\top\lambda = g\).
4. A full `jax.vjp` of the residual / coupling assembly (`_merged_constraint`), which re-runs `set_params` + SupportBeams `support_values` / `coupling_values` while (1)–(3) are still live.

There is **no** `jax.checkpoint` / rematerialization anywhere under `src/coil_fem/`.

**API caveat:** `CoilFEMObjective.J()` and `dJ()` both call the same `jax.value_and_grad` JIT (`_jit_vg`). A true forward-only path is `CoilFEM.run` / `CoilFEM.objective` without `grad`. If measured `J()` ≪ `dJ()`, that comparison is likely vs a forward-only solve, vs XLA DCE of unused grads, or vs allocator peak timing — not a separate cheap objective compile inside `CoilFEMObjective`.

---

## 1. Call graph

### Forward (value)

```
CoilFEMObjective.J() / dJ()
  └─ _compute_J / _compute_dJ
       └─ self._jit_vg(cdofs, idofs, sdofs)   # jit(value_and_grad(_weighted_J))
            └─ _weighted_J
                 └─ CoilFEM.objective(...)
                      ├─ _expand_geometry
                      ├─ _solve_all(...)
                      │    ├─ SupportBeams.beam_geometry          # weights path
                      │    ├─ per coil:
                      │    │    mesh_points_from_dofs
                      │    │    _jit_fe_geom_fns → (sg, jxw, vgj, pqp)
                      │    │    _jit_body_force_fns → (bf, B_self, B_ext)
                      │    │    support.stiffness(*compute_weights(...))
                      │    └─ solve_monolithic → static.merged_solve(...)
                      │         └─ make_merged_solve custom_vjp _fwd
                      │              ├─ _assemble_merged_values
                      │              │    surface_quad_points / surface_jxw
                      │              │    beam_geometry (again)
                      │              │    pipeline.assemble_coo  # set_params → V_jax
                      │              │    support_values / coupling_values
                      │              ├─ assemble_csr_values(V_merged)
                      │              └─ solver_K(f, csr_values)
                      └─ metrics(sol, sg, jxw)
```

### Reverse (`value_and_grad` / `dJ`)

```
∂J/∂sol → merged_solve._bwd(res, g_flat)
  ├─ assemble_csr_values(V_merged, coo_to_csr_T)   # reuse V, not reassemble K
  ├─ solver_KT(g_flat, csr_T) → λ                   # SECOND factorization
  ├─ jax.vjp(_merged_constraint, bcd, sdofs, pts, bf, k)
  │    constraint re-runs:
  │      set_params + compute_residual_vars (per coil)
  │      support_values + coupling_values (beam_geometry recomputed from sdofs)
  └─ outer reverse through:
       bf ← B_self / B_ext / Lorentz / biot_savart
       pts ← mesh + FE geom
       k  ← compute_weights / attachment_fn
       metrics ← shape_grads / stress
```

Key files:

| File | Role |
|------|------|
| `src/coil_fem/simsopt/objectives.py` | `J` / `dJ` → `_jit_vg` |
| `src/coil_fem/coil_fem.py` | `objective`, `_solve_all`, `build_monolithic_static` |
| `src/coil_fem/coupling/drivers.py` | `make_merged_solve` `_fwd`/`_bwd`, `solve_monolithic` |
| `src/coil_fem/coupling/beam_network.py` | `beam_geometry`, `compute_weights`, `coupling_values`, `support_values` |
| `src/coil_fem/solvers/cudss.py` | CSR assemble, cuDSS factor/solve |
| `src/coil_fem/pipelines.py` | `assemble_coo` → Jacobian + load |
| `src/coil_fem/problems/device_problem.py` | On-device `V_jax` |
| `src/coil_fem/problems/linear_elasticity.py` | `recompute_fe_geometry`, `set_params` |
| `src/coil_fem/magnetic.py` | Biot–Savart / self-field |
| `src/coil_fem/metrics.py` | Post-solve reverse through stress |

---

## 2. AD patterns (monolithic vs staggered)

| Path | Status | AD mechanism |
|------|--------|----------------|
| **Monolithic** (`coupling='monolithic'`, `solver='cudss'`) | Active | `make_merged_solve` → `@jax.custom_vjp` with IFT-style residual VJP: \(K^\top\lambda=g\), then \(\partial r/\partial p\) |
| **Per-coil cuDSS** (`cudss_ad_wrapper`) | Uncoupled only | Same idea: `custom_vjp` + adjoint solve + `vjp(constraint_fn)` |
| **Staggered** (`solve_staggered`) | Retired / `NotImplementedError` | Docs describe fixed-point IFT; not available |

Monolithic **does** implement custom VJP / IFT. It does **not** reuse staggered’s fixed-point IFT. Peak-memory design notes for monolithic vs staggered live in `notes/SUPPORT_PLANS.md` (combined factor ≥ sum of parts; ~\(O(N^{4/3})\) fill in 3D).

---

## 3. Concrete memory amplifiers

### A. Outer `value_and_grad` tape (alive across the solve)

Saved so reverse can reach curve / current / support DOFs even though `merged_solve` has its own VJP:

| Intermediate | Where | Why large |
|--------------|-------|-----------|
| `shape_grads` `(n_cells, n_quads, n_nodes, 3)` | `_jit_fe_geom_fns` / `recompute_fe_geometry` | Dominant dense FEM tensor; TET10 ≫ TET4 |
| `v_grads_JxW`, `pqp` | same | Same order |
| `B_self`, `B_ext`, `bf` `(n_cells, n_quads, 3)` each | `_body_force_at_quads` | Held so `g_bf` backprops through Biot–Savart / self-field |
| Surface weight / attachment intermediates | `compute_weights` → `_clamp_weights_for_spec` → `attachment_fn` | Per endpoint × `n_surface_quads` |
| Metric tensors | `von_mises_on_quadrature` | `u_grad`, `sigma` at all quads |

`objective` only needs `sol` + `sg`/`jxw` for metrics, but `bf`/`pts`/`k` are inputs to `merged_solve`, so their producers stay on the tape.

### B. `custom_vjp` residual (explicitly stashed in `_fwd`)

From `coupling/drivers.py`:

```python
return sol_flat, (bcd, sdofs, pts, bf, k, fe_geom, sol_flat, geom, V_merged)
```

| Residual leaf | Role |
|---------------|------|
| `V_merged` | Full merged COO values (`K_cc` + `K_ss` + `K_cs`/`K_sc`) — largest single AD residual buffer |
| `pts`, `bf`, `k` | Full per-coil mesh / force / stiffness (duplicated vs outer producer outputs) |
| `geom` | Outer forward-cache geom; residual stash is for zero cotangent only — `_bwd` recomputes `beam_geometry` |
| `sol_flat` | Length `n_total_dofs` (all coils + beams) |

Forward-only would not need `V_merged` after the solve. Reverse keeps it until `_bwd` finishes.

### C. Two cuDSS workspaces + second factor

`CoilFEM.build_monolithic_static` builds **both** `solver_K` and `solver_KT` and holds them on `MonolithicStatic` for the object lifetime.

- Forward: `solver_K(f, csr_values)` — numeric factor of merged \(K\).
- Backward: `solver_KT(g, csr_values_T)` — **separate** factor of \(K^\top\), not a solve-with-cached-factor of \(K\).

For SPD systems \(K^\top = K\), this is a large VRAM duplicate of the dominant cost.

### D. `_bwd` re-traces a heavy constraint under `jax.vjp`

`_merged_constraint` re-runs per coil:

- `set_params` → full `recompute_fe_geometry` + surface geometry (**does not** receive precomputed `_fe_geom`)
- `compute_residual_vars`
- `support_values` + `coupling_values` (with **recomputed** `beam_geometry` from `sdofs`)

That AD tape peaks **while** outer intermediates and `V_merged` are still live.

### E. Duplicate forward work that also fattens the tape

1. **`beam_geometry` twice:** `_solve_all` (weights) and `_assemble_merged_values` (assembly).
2. **`recompute_fe_geometry` again inside `assemble_coo`:** `_solve_all` already builds `fe_geom_by_coil` and puts it in `driver_params`, but `solve_monolithic` never passes it into `merged_solve`; `_coil_params` / `assemble_coo` omit `_fe_geom`, so `set_params` recomputes. Same recompute happens again in `_bwd`.
3. **Surface quads / JxW again** inside `_assemble_merged_values` after weights already used `surface_quad_points`.
4. **`coupling_values` / `support_values`:** per endpoint, `einsum` over all surface quads → node-folded dense `3×3` blocks (`skew_eff` `(n_surf_nodes, 3, 3)`), then flatten into COO.

### F. `beam_geometry` in `_bwd` — must recompute (not freeze)

**Incorrect (do not reintroduce):**

```python
geom_in = fwd_geom  # freezes ∂K/∂φ through beam frame — breaks SupportBeams dJ
```

Freezing the forward-cached geom inside the constraint VJP was tried as a memory win. It cuts \(\partial K/\partial\phi\) through endpoints / `gamma3` / `L_eff` into `K_ss`/`K_cs`/`K_sc`, so analytic support gradients disagree with FD (Taylor test: `|dJh| ≫` centered difference; L-BFGS stalls). Partial flow through outer Winkler `k` is not enough.

**Required:**

```python
geom_in = support.beam_geometry(curves, sdofs_in)  # differentiate through geom
```

Forward-only sharing of `support_geom` in `_solve_all` / `_assemble_merged_values` remains valid. The outer `geom` argument cotangent stays zero; support-DOF grads flow via `sdofs → beam_geometry` inside the constraint. Allowed memory tool: `jax.checkpoint` / remat on that recompute — **never** freeze geom.

---

## 4. Locations that dominate memory

Ranked roughly by peak impact for monolithic + SupportBeams + cuDSS:

1. **`solver_K` + `solver_KT`** — `build_monolithic_static` / `make_merged_solve` — two sparse-direct factors of the merged system.
2. **`V_merged` in custom_vjp residual** — `_fwd` residual tuple — full COO of merged \(K\).
3. **`recompute_fe_geometry` / `shape_grads`** — called from `_solve_all`, again in `assemble_coo`/`set_params`, again in `_bwd` constraint.
4. **`DeviceProblem.compute_newton_vars` → `V_jax`** — per-coil cell + face Jacobian flats concatenated into merged COO.
5. **`_body_force_at_quads` / `biot_savart` / `B_self_quadrature`** — large `(n_cells, n_quads, 3)` fields on the outer AD tape.
6. **`SupportBeams.coupling_values` + `support_values`** — surface-quad → node folds, `12×12` spring blocks, dense `skew` tensors per endpoint.
7. **`jax.vjp(_merged_constraint)`** — second assembly-scale AD region stacked on residual + outer tape.
8. **Metrics** — `von_mises_on_quadrature` — stress at all quads (smaller than K / FE geom, still non-trivial).

---

## 5. Forward `J()` vs reverse `dJ()` profile

### Ideal forward-only (`fem.run` / `objective` without grad)

Live: mesh → FE geom → B/forces → assemble COO → **one** cuDSS factor → sol → metrics.

Release after use: assembly temps; no `V_merged` residual; no `solver_KT`; no constraint VJP; no outer Biot–Savart reverse tape.

### Actual `CoilFEMObjective.J()` / `dJ()`

Both go through `_jit_vg = jit(value_and_grad(_weighted_J))` when `solver=='cudss'`:

```python
# simsopt/objectives.py
_vg = value_and_grad(self._weighted_J, argnums=(0, 1, 2))
self._jit_vg = jax.jit(_vg)

def _compute_J(self):
    J_val, _ = self._jit_vg(...)   # discards grads; docstring claims "without an adjoint"

def _compute_dJ(self):
    J_val, grads = self._jit_vg(...)
    # materializes host grad arrays
```

| | Forward-only objective | `value_and_grad` (`J`/`dJ` as coded) |
|--|------------------------|--------------------------------------|
| cuDSS factors | 1 (`solver_K`) | up to 2 (`K` then `KT`) |
| Keep `V_merged` after solve | no | yes (residual) |
| Outer FE / Biot tape | discard after use | keep until reverse completes |
| Constraint `vjp` | no | yes |
| Host grad arrays | no | yes in `_compute_dJ` |

Also: calling `J()` then `dJ()` can run `_jit_vg` **twice** (independent `_needs_J` / `_needs_dJ` flags), doubling wall time and stressing the allocator. `_compute_dJ` already refreshes the `J` cache; the reverse order is cheaper.

---

## 6. SupportBeams-specific pressure

- **`beam_geometry`:** endpoints, `gamma3`, `L_eff` — moderate size; **must** be recomputed under AD in `_bwd` (freezing is incorrect; use checkpoint/remat for memory).
- **`compute_weights` / `_clamp_weights_for_spec`:** every coil-touching endpoint maps **all** surface quads into beam frame and calls `attachment_fn` — scales as `n_endpoints × n_surface_quads`.
- **`coupling_values`:** builds `skew_sq` `(n_sel, n_fq, 3, 3)`, folds to `skew_eff` `(n_surf_nodes, 3, 3)`, emits dense `3×3` blocks into COO — interface nnz can be large even if beams are few.
- **`support_values`:** `N_beams × 144` COO from local + spring `12×12`; spring path again sums over surface quads per endpoint.
- Monolithic couples **all** base coils into one system — one factor over \(\sum_i n_{\mathrm{dofs},i} + n_s\).

---

## 7. Actionable tips (by likely impact)

### High leverage (code / API)

1. **True forward-only `J()`**  
   Compile/cache `jax.jit(self._weighted_J)` for `_compute_J`; keep `_jit_vg` only for `_compute_dJ`. Biggest easy win if callers use `J()` alone or expect cheap value checks. Aligns `_compute_J`’s docstring with reality.

2. **Reuse one cuDSS factor for the adjoint**  
   For SPD merged \(K\) (\(K^\top = K\)), prefer solve-only with `solver_K` for \(\lambda\), or a spineax warm/refactor API, instead of a second live `solver_KT` workspace. Largest VRAM lever after system size.

3. **`jax.checkpoint` on the outer pre-solve**  
   Rematerialize `_jit_fe_geom_fns` and especially `_jit_body_force_fns` (`biot_savart` / `B_self`) in reverse. Trades compute for peak memory on the outer tape. Nothing does this today.

4. **Pass `fe_geom` into monolithic assemble**  
   Thread `fe_geom_by_coil` from `driver_params` into `_coil_params` / `assemble_coo` / `_bwd` `set_params` so FE geometry is not recomputed two extra times. Cuts both time and AD intermediates. Infrastructure already exists for uncoupled (`solve_uncoupled` + `pipeline.solve(..., fe_geom=...)`); monolithic just does not use it.

5. **Avoid double `_jit_vg`**  
   If both `J` and `dJ` are needed, prefer `dJ` first (or have `_compute_J` share results when grads were already computed). Today `J()` then `dJ()` recompiles/reruns the full `value_and_grad` path.

### Medium leverage

6. **Don’t stash full `V_merged` if residual+factor peaks dominate**  
   Residual currently avoids reassembly in `_bwd`. Optionally rematerialize `V` in `_bwd` and drop it from the residual if peak RAM is residual-bound (trade: more `_bwd` compute). Profile which of residual vs factor dominates first.

7. **Checkpoint or slim `_merged_constraint` VJP**  
   Remat `set_params`+residual and/or SupportBeams `coupling_values`; avoid materializing full `skew_eff` denser than needed.

8. **Mesh / element downgrade for profiling / large runs**  
   TET4 vs TET10, fewer `n_phi` / cells: shrinks `shape_grads`, `V_jax`, and factor fill together.

9. **Share `beam_geometry` across weights + assemble**  
   Pass the `_solve_all` geometry into `_assemble_merged_values` so the second JIT call and its intermediates disappear from the outer tape.

### Lower leverage / later

10. **dtype** — Stack is float64 (`jax_enable_x64`). float32 would cut ~2× buffers/factor but needs careful numerics for SPD/direct solve — high risk.

11. **Batching / multi-GPU** — Not supported on this monolithic path; splitting coils only helps uncoupled. Staggered is retired (see `SUPPORT_PLANS.md` for why staggered was better for peak memory when available).

12. **Host materialization** — `_compute_dJ` copies grads to numpy (`np.asarray`). Small vs GPU factor, but avoid holding both device and host copies longer than needed.

---

## 8. Suggested measurement plan

Before changing code, confirm where the peak actually sits:

1. Compare three peaks on the same DOFs:
   - `fem.objective(...)` (no grad)
   - `CoilFEMObjective.J()` (current `_jit_vg`, discard grads)
   - `CoilFEMObjective.dJ()`
2. Inside `dJ`, instrument (or temporarily stub) to isolate:
   - forward factor only (`solver_K`)
   - residual footprint (`V_merged` size = `nnz_coo * 8` bytes)
   - adjoint factor (`solver_KT`)
   - constraint `vjp` alone
3. Report `jax.local_devices()[0].memory_stats()` (or `nvidia-smi`) at those points; nnz of merged COO / CSR is already available on `MonolithicStatic`.

Expected pattern if this audit is right:  
`objective` ≪ `J()` (if XLA still builds the grad graph) ≪ `dJ()`, with the largest step from `J`/`objective` → `dJ` coming from `solver_KT` + residual + outer Biot/FE tape.

---

## 9. Bottom line

Monolithic + SupportBeams already pays for one large GPU factor in forward. **`dJ()` adds** (a) outer reverse through FE geometry + Lorentz/Biot fields, (b) residual retention of **`V_merged` + inputs**, (c) a **second** transposed cuDSS factor, and (d) a full **`jax.vjp` of residual + beam coupling assembly**. That stack explains substantially higher peak memory than a forward-only solve.

Highest-leverage levers, in order:

1. Stop routing `J()` through `value_and_grad`.
2. Collapse `solver_KT` into a reuse of the forward factor (SPD).
3. Checkpoint body-force / FE geom on the outer tape.
4. Thread `fe_geom_by_coil` into monolithic assemble / `_bwd` (already computed, currently unused there).
