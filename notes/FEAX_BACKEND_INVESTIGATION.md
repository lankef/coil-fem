# FEAX Backend Investigation

Investigation into adding [FEAX](https://github.com/Naruki-Ichihara/feax) as a
second FEM solver backend alongside the existing JAX-FEM backend, sharing the
same `meshing.py` mesh infrastructure.

## Executive Summary

Adding FEAX is **feasible without heavy rework**.  The two libraries share the
same lineage (FEAX acknowledges JAX-FEM as a reference implementation) and expose
nearly identical `Problem` subclass APIs.  The main cost is a thin adapter layer
for mesh format, boundary conditions, internal variables, and solver invocation.
The constitutive law (`get_tensor_map`) and surface maps
(`get_surface_maps`) can be shared essentially verbatim.

---

## 1. Steps Required

### 1.1 Mesh Adapter

`CoilMesh` is now an abstract base (`abc.ABC`) that inherits from
`jax_fem.generate_mesh.Mesh`, with two concrete subclasses `CoilMeshRectangle`
and `CoilMeshDisk` that own their cross-section metadata and implement
`mesh_points_from_dofs`.  FEAX's `feax.mesh.Mesh` is a plain container with the
same three base fields (`points`, `cells`, `ele_type`).

**Work:**

- Write a `to_feax_mesh(coilmesh) -> feax.mesh.Mesh` converter.  This is a
  one-liner (`feax.mesh.Mesh(coilmesh.points, coilmesh.cells, coilmesh.ele_type)`)
  because both use the same field names and element-type strings (`TET4`,
  `TET10`, `HEX8`).
- Alternatively, make `CoilMesh` backend-agnostic by inheriting from neither and
  providing `.to_jaxfem()` / `.to_feax()` helpers.  This is cleaner long-term.

### 1.2 Boundary Condition Translation

JAX-FEM uses a list-based format: `[location_fns, vecs, value_fns]`.
FEAX uses dataclasses: `DirichletBCSpec(location, component, value)` assembled
into a `DirichletBCConfig` that produces a `DirichletBC` pytree.

**Work:**

- Write a shared BC specification format (or use FEAX's `DirichletBCSpec` as the
  canonical format since it is more expressive) and convert to JAX-FEM's format
  when needed.
- The existing `dirichlet_bc()` helper in `elasticity.py` already takes a
  `selection_rule` and `value`; refactoring it to also emit FEAX BCs is
  straightforward.
- FEAX location functions operate on a single point (`(dim,) -> bool`) whereas
  the current JAX-FEM helpers evaluate on a batch `(N, 3) -> (N,)` and close
  over the result.  The adapter must wrap or `vmap` appropriately.

### 1.3 Problem Class

Both libraries use the same subclass pattern:

| Concept | JAX-FEM (`elasticity.py`) | FEAX |
|---------|--------------------------|------|
| Constitutive law | `get_tensor_map() -> stress(u_grad, *args)` | `get_tensor_map() -> stress(u_grad, *args)` |
| Body force | `get_mass_map() -> mass_map(u, x, bf)` | `get_mass_map() -> mass_map(u, x, *internal_vars)` |
| Surface loads | `get_surface_maps() -> [f(u, x)]` | `get_surface_maps() -> [f(u, x, *internal_vars)]` |
| Energy density | N/A | `get_energy_density() -> psi(u_grad, *args)` |
| Geometry update | `set_params(params)` (custom Path C) | Not needed (geometry lives in mesh, a static pytree aux) |

**Work:**

- Factor the stress law (Hooke's law with optional thermal eigenstrain) into a
  standalone function that both Problem subclasses call.  This already half-exists
  in `metrics.py::cauchy_stress_small_strain` and
  `problem/linear_elasticity.py::LinearElasticity3D.get_tensor_map`
  (with the thermal eigenstrain from `problem/linear_elasticity.py::itc_strain`).
- Write `LinearElasticity3D_FEAX(feax.problem.Problem)` mirroring the existing
  `LinearElasticity3D(jax_fem.problem.Problem)`.
- The FEAX subclass will be simpler: no `set_params` override is needed because
  FEAX's `Problem` is a frozen dataclass pytree—mesh points are already static
  aux data.  Differentiability through mesh node positions follows a different
  path (see 1.5 below).

### 1.4 Internal Variables / Loads

JAX-FEM passes body force through `internal_vars` (a list of arrays
set in `set_params`).  FEAX uses `InternalVars`, a frozen-dataclass pytree with
`volume_vars` and `surface_vars` tuples.

**Work:**

- Write a helper `body_force_to_feax_internal_vars(problem, bf_array)` that wraps
  the Lorentz body-force array into FEAX's `InternalVars` format.
- FEAX `InternalVars` supports node-based, cell-based, or quad-point-based
  variables with automatic interpolation—this actually simplifies the interface
  compared to JAX-FEM where the caller must pre-evaluate at quad points.

### 1.5 Solver Invocation and Differentiability

JAX-FEM differentiability path ("Path C" in `elasticity.py`):

1. `ad_wrapper(problem)` returns a function `fwd_pred(params) -> sol`.
2. Inside `fwd_pred`, `set_params(params)` recomputes all FE geometry arrays
   from `params['points']` via pure JAX, enabling `jax.grad` through mesh node
   positions.

FEAX differentiability path:

1. `create_solver(problem, bc, ..., internal_vars=iv)` returns a callable
   `solver(internal_vars, initial) -> sol`.
2. `solver` has a custom VJP; gradients flow through `internal_vars` (body
   force, material properties) automatically.
3. Geometry differentiation (through mesh node positions) works differently:
   since `Problem` is a frozen pytree whose mesh is static aux data, you would
   need to rebuild the Problem for each new set of points.  This is fine for
   forward solves but means geometry AD requires a different strategy than
   JAX-FEM's `set_params`.

**Work:**

- For **load/current differentiation** (the primary use case): FEAX's
  `create_solver` → `solver(internal_vars)` with `jax.grad` works out of the
  box.  Just pass Lorentz body force as an `InternalVars` volume variable.
- For **geometry (coil DOF) differentiation**: Two options:
  - **Option A (recommended):** Compute mesh points as a differentiable function
    of coil DOFs (this is what `meshing.py` already does), then rebuild the FEAX
    Problem inside a `jax.grad`-traced function.  Since FEAX's `Problem.__post_init__`
    does heavy index computation, this may be expensive at compile time but is
    correct.
  - **Option B:** Port `recompute_fe_geometry` to work with FEAX's internal
    arrays (shape_grads, JxW, etc.).  This is more invasive and couples to FEAX
    internals.
- Write a unified `solve_elasticity(mesh, bc, body_force, E, nu, backend="jaxfem"|"feax")`
  top-level function that dispatches to the appropriate solver.

### 1.6 Post-Processing / Objectives

`fem_objectives.py` computes von Mises stress, strain energy, etc. by accessing
`problem.fes[0].cells`, `problem.fes[0].shape_grads`, `fe.JxW`, etc.  The field
names are identical in FEAX (`problem.fes[0].cells`, `.shape_grads`, `.JxW`),
so most of `fem_objectives.py` should work with both backends unmodified.

**Work:**

- Verify that array shapes match (they should—both use
  `(num_cells, num_quads, num_nodes, dim)` layout).
- Minimal adapter if FEAX stacks arrays differently for multi-variable problems
  (unlikely for our single-variable elasticity case).

### 1.7 Thermal Eigenstrain

`thermal.py` is backend-agnostic: it returns stress tensors from `(u_grad, lam, mu, epsilon_th)`.  The only coupling point is how `epsilon_th` enters the
Problem's `get_tensor_map`.

**Work:**

- Both backends support passing extra data through internal variables; the
  thermal eigenstrain tensor can ride alongside the body force.
- Factor out a shared `stress_with_thermal(u_grad, lam, mu, epsilon_th)` that
  both Problem subclasses call from their `get_tensor_map`.

### 1.8 Packaging and Optional Dependency

**Work:**

- Add `feax` as an optional dependency group in `pyproject.toml`:
  `[project.optional-dependencies] feax = ["feax"]`.
- Guard FEAX imports with `try/except ImportError` and a `_HAS_FEAX` sentinel,
  following the existing `_HAS_JAXFEM` pattern.
- Both backends can coexist: a user installs `pip install -e ".[fem]"` for
  JAX-FEM, `pip install -e ".[feax]"` for FEAX, or both.

---

## 2. Key Challenges

### 2.1 Geometry Differentiation (Coil DOF → Mesh → Solution)

This is the **hardest part**.  JAX-FEM's `set_params` / `ad_wrapper` mechanism
lets `jax.grad` trace through mesh node positions because the FE geometry
(shape_grads, JxW) is recomputed from `points` inside the traced function.

FEAX treats `Problem` as a frozen dataclass whose mesh is static aux data.
Rebuilding the Problem for each new geometry works but triggers
re-computation of all FE index arrays and shape function evaluations in
`__post_init__`.  This is:
- **Correct** under `jax.grad` (the index arrays are static/integer and don't
  carry gradients; shape_grads and JxW do and are recomputed).
- **Potentially expensive** at JIT compile time if the mesh topology changes
  shape (it won't for our fixed-topology meshes, so this is likely fine).

**Mitigation:** For fixed-topology meshes, the integer connectivity arrays
never change.  Only `points` changes.  We can pre-build the Problem once for
topology, then override `fes[i].shape_grads` / `JxW` inside a traced function,
similar to what `recompute_fe_geometry` does for JAX-FEM.  This is the most
efficient path but couples to FEAX internals.

### 2.2 Solver Behavior Differences

| Aspect | JAX-FEM | FEAX |
|--------|---------|------|
| Linear solve | `scipy.sparse.linalg` (UMFPACK) or iterative | cuDSS (GPU direct), spsolve, CG, BiCGSTAB, GMRES |
| Newton | Custom while-loop | Fixed-iter or adaptive while-loop with line search |
| Matrix format | Custom sparse | JAX `BCOO` sparse |
| Custom VJP | `ad_wrapper` (external) | Built into `create_solver` |

The different solver numerics (direct vs iterative, UMFPACK vs cuDSS) will
produce slightly different solutions due to floating-point ordering.  For
validation/benchmarking this is a feature, not a bug.

### 2.3 FEAX is Newer and Less Battle-Tested

FEAX has ~80 GitHub stars and is under active development.  API breakage
between versions is more likely than with JAX-FEM.  Pin the FEAX version in
`pyproject.toml`.

### 2.4 JAX Version Compatibility

FEAX requires JAX 0.7+, while `pyproject.toml` currently pins `jax >= 0.6.2`.
This is a hard conflict.  Options:
- Upgrade the project to JAX 0.7+ (may require changes elsewhere).
- Keep JAX-FEM on >=0.6.2 and make FEAX available only when JAX >= 0.7 is
  installed.
- Wait for JAX-FEM to also support 0.7+.

### 2.5 Winkler (Spring) Boundary Conditions

The existing JAX-FEM backend supports Winkler BCs via `get_surface_maps` and
`location_fns`.  FEAX also supports surface maps with the same pattern but
through its own `InternalVars` surface variables.  The translation is
straightforward but must be implemented.

---

## 3. Proposed Code Architecture Changes

### 3.1 Current Structure

```
elasticity.py          # JAX-FEM Problem + solver + post-processing + BC helpers
fem_objectives.py      # Von Mises / strain energy (accesses jax_fem internals)
thermal.py             # Thermal eigenstrain (backend-agnostic)
meshing.py             # CoilMesh inherits jax_fem.generate_mesh.Mesh
simsopt_bridge.py      # Optimizable wrappers
```

### 3.2 Proposed Structure

```
meshing.py             # CoilMesh becomes backend-agnostic (no jax_fem inheritance)
                       #   .to_jaxfem() -> jax_fem.generate_mesh.Mesh
                       #   .to_feax()   -> feax.mesh.Mesh

material.py (NEW)      # Shared material laws:
                       #   lame_parameters(), stress_hooke(), stress_with_thermal()
                       #   Backend-agnostic, pure JAX functions.

boundary.py (NEW)      # Unified BC specification:
                       #   CoilBC dataclass (location_rule, component, value)
                       #   .to_jaxfem(mesh) -> [loc_fns, vecs, val_fns]
                       #   .to_feax(problem) -> DirichletBC

elasticity.py          # Refactored:
                       #   LinearElasticity3D (JAX-FEM, as today)
                       #   recompute_fe_geometry (stays, JAX-FEM specific)
                       #   solve_linear_elasticity (JAX-FEM specific)
                       #   Visualization helpers (stay)

elasticity_feax.py     # NEW: FEAX backend
  (NEW)                #   LinearElasticity3D_FEAX(feax.problem.Problem)
                       #   solve_linear_elasticity_feax()
                       #   Calls shared material laws from material.py

fem_objectives.py      # Mostly unchanged; works with both backends since
                       #   field names (fes, shape_grads, JxW, cells) match

thermal.py             # Unchanged (already backend-agnostic)

solver_dispatch.py     # NEW (optional): Unified entry point
  (NEW)                #   solve(mesh, bc, body_force, E, nu, backend=...)
                       #   Returns (solution, problem) for post-processing
```

### 3.3 Shared Infrastructure Summary

| Component | Shared? | Notes |
|-----------|---------|-------|
| Meshing (`CoilMesh`, sweeps) | Yes | Backend-agnostic after removing jax_fem inheritance |
| Material laws (Hooke, thermal) | Yes | Factor into `material.py` |
| BC specification | Yes | Unified `CoilBC` with backend converters |
| Constitutive `get_tensor_map` body | Yes | Both call the same `stress()` closure |
| Post-processing (`fem_objectives.py`) | Mostly | Same field names; minor shape checks |
| Solver invocation | No | Different APIs, different AD mechanisms |
| Geometry AD (`recompute_fe_geometry`) | No | JAX-FEM specific; FEAX needs its own approach |

---

## 4. Difficulty Assessment

| Task | Difficulty | Reason |
|------|-----------|--------|
| Mesh adapter | Trivial | Same fields, same naming |
| BC translation | Low | Mechanical format conversion |
| FEAX Problem subclass | Low | Nearly identical API to JAX-FEM Problem |
| Load/current differentiability | Low | FEAX `create_solver` supports this natively |
| Post-processing reuse | Low | Same array layouts |
| Thermal eigenstrain | Low | Already backend-agnostic |
| Geometry (coil DOF) differentiability | **Medium** | Requires strategy decision; either rebuild Problem per-call or port `recompute_fe_geometry` for FEAX |
| JAX version conflict (0.6 vs 0.7) | **Medium** | May require project-wide JAX upgrade |
| Refactoring `CoilMesh` to be backend-agnostic | Low | Remove one inheritance, add two converters |
| Testing both backends | Low-Medium | Need parallel test fixtures for same-input / compare-output |

**Overall assessment:** This is a moderate-scope refactoring, not a rewrite.
The two libraries are similar enough that most of the work is plumbing (format
conversion, optional imports) rather than algorithmic.  The only genuinely
non-trivial piece is the geometry differentiation strategy for the FEAX path,
and even that has a clear (if slightly expensive) solution: rebuild the Problem
inside the traced function for fixed-topology meshes.
