# AGENTS.md

Guidance for AI coding agents working on the **coil-fem** repository.

## Environment Setup

- **Conda env:** `rod` (Python 3.12). Activate with `conda activate rod`.
- **Install:** `pip install -e ".[dev]"` (editable, with test/notebook extras).
- **Optional extras:** `pip install -e ".[docs]"` (Sphinx).
- **simsopt** is installed from a local editable checkout at `../simsopt` — it is *not* declared in `pyproject.toml`.
- **stellcoilbench** is installed from a local editable checkout at `../stellcoilbench`.

### GPU cuDSS solver extra (`.[cudss]`)

The `cudss` extra installs the GPU sparse direct solver stack (spineax +
NVIDIA cuDSS) used by `problem_options={'solver': 'cudss'}`. Because spineax
compiles CUDA at install time, it needs a real `nvcc` and `--no-build-isolation`:

```bash
# 1. A real nvcc matching the CUDA 12.9 runtime. The pip `nvidia-cuda-nvcc-cu12`
#    wheel ships only `ptxas`, so install nvcc via conda:
conda install -c conda-forge cuda-nvcc=12.9.86

# 2. Build spineax against the installed jaxlib/XLA headers and install the extra:
pip install --no-build-isolation -e ".[cudss]"
```

Caveats:

- `--no-build-isolation` is required (spineax's CMake locates the installed
  `jaxlib`/XLA FFI headers; an isolated build env would not have them).
- `nvidia-cudss-cu12` is pinned `<0.8`. cuDSS 0.8 is a breaking API change
  (`cudaDataType_t` → `cudssDataType_t`, plus a new `offsetType` argument to
  `cudssMatrixCreateCsr`) that spineax does not yet support. Do not let it
  upgrade to ≥0.8 or the build will fail to compile.
- `spineax` is pulled from `git+https://github.com/johnviljoen/spineax.git`
  (not on PyPI under that name), so the project is not PyPI-publishable as-is.

## Running Tests

```bash
pytest                                    # full suite
pytest tests/test_kirchhoff.py            # single file
pytest tests/test_kirchhoff.py::test_name # single test
```

pytest is configured via `[tool.pytest.ini_options]` in `pyproject.toml` with `testpaths = ["tests"]`.

## Project Layout

```
src/coil_fem/                  # main package (Hatchling src-layout)
  __init__.py                  # re-exports CoilFEM, biot_savart, B_self_quadrature, lorentz_body_force
  coil_fem.py                  # CoilFEM — differentiable FEM pipeline container
  magnetic.py                  # B-field helpers (biot_savart, B_self_quadrature) + lorentz_body_force
  metrics.py                   # Von Mises / strain metrics on FEM solutions
  meshing.py                   # Fixed-topology hex/tet meshing (rectangle/disk sweep, curved-sided TET10)
  pipelines.py                 # ElasticPipeline / ThermoElasticPipeline — per-coil FEM state
  problems/                    # FEM Problem subpackage
    __init__.py                # re-exports LinearElasticity3D, DeviceProblem (+ elasticity helpers)
    linear_elasticity.py       # LinearElasticity3D — JAX-FEM Problem subclass; itc_strain thermal eigenstrain; support_attach shifted Winkler
    device_problem.py          # DeviceProblem — JAX device-assembly Problem subclass
    heat_conduction.py         # HeatConduction3D stub (future thermoelastic coupling)
  presets/                     # Named material / cross-section factory helpers
    __init__.py
    cross_section_fns.py       # solid/hollow circle & rectangle section factories
  geo/                         # Curve geometry and symmetry subpackage
    __init__.py                # re-exports CurveXYZFourierJAX, framed curves, symmetry helpers
    curve_jax.py               # CurveXYZFourierJAX — JAX pytree, simsopt interop
    framed_curve_jax.py        # FramedCurveCentroidJAX / FramedCurveRMFJAX
    symmetries.py              # Stellarator symmetry expansion (pure JAX)
  coupling/                    # Support structure coupling subpackage
    __init__.py                # re-exports Support, SupportBeams, solve_staggered, solve_monolithic
    supports.py                # Support (concrete grounded Winkler/Robin BC)
    beam_networks.py           # SupportBeams — bisymmetric beam-network support (coil-coil + coil-foundation)
    drivers.py                 # solve_staggered (BG-S + Aitken + IFT grad), solve_monolithic (cuDSS-only)
  simsopt/                     # simsopt Optimizable interop subpackage
    __init__.py                # re-exports CoilFEMObjective, CoilSupport, CoilSupportFixed, CoilSupportTopBottom
    objectives.py              # CoilFEMObjective — simsopt Optimizable wrapper
    optimizables.py            # CoilSupport (base), CoilSupportFixed, CoilSupportTopBottom
  solvers/                     # Optional GPU solver subpackage
    __init__.py
    cudss.py                   # GPU sparse direct solver (spineax + NVIDIA cuDSS)
pyproject.toml                 # Hatchling build, deps, pytest config
examples/                      # Runnable workflow scripts
docs/                          # Sphinx documentation (conf.py, RST, tutorials)
tests/                         # pytest tests
data/                          # Coil geometry files (gitignored)
```

## Key Dependencies

| Package | Purpose | Notes |
|---------|---------|-------|
| `jax >= 0.6.2` | Autodiff, JIT, vmap | Core dependency |
| `lineax` | Differentiable linear solvers | Core dependency |
| `simsopt` | Magnetic forces, coil geometry | Local source `../simsopt`, not in pyproject.toml |
| `jax-fem` | FEM analysis | Core dependency |
| `meshio` | Mesh I/O | Core dependency |

## Coding Conventions

### Imports and Style

- Use `from __future__ import annotations` for deferred type evaluation in modules that need forward references.
- Prefer `jax.numpy as jnp` for traced numeric code; use bare `numpy` only for compile-time constants (mesh connectivity tables, etc.).
- Type hints: use `jax.Array | float` union style for function signatures.

### JAX Patterns

- **Pytrees:** Register custom classes with `@jax.tree_util.register_pytree_node_class` and implement `tree_flatten` / `tree_unflatten`. Traced leaves are arrays; static data (e.g. `order`) goes in `aux_data`.
- **JIT/vmap:** Use `jax.jit` and `jax.vmap` for vectorized operations. Prefer `functools.partial` for binding static args.
- **Optional heavy deps:** Guard imports with `try/except ImportError` and set a `_HAS_*` sentinel if a dependency is truly optional.

### Docstrings and Comments

#### Module docstrings

Every `.py` file must start with a module docstring. Format: one-sentence summary on the opening `"""` line, followed by an optional short paragraph (≤3 sentences) of **user-facing** context — what the module provides and when to use it. No internal design rationale, no change history, no step-by-step descriptions of the implementation.

```python
"""Structured volume meshes for coil cross-sections.

Sweeps a rectangular or disk cross-section grid along a framed curve to
produce a ``CoilMesh`` (TET4 or TET10) used by :class:`~coil_fem.CoilFEM`.
"""
```

#### Function and class docstrings

Use **NumPy style** throughout. The one-line summary goes on the same line as the opening `"""`. Use `Parameters`, `Returns`, `Raises`, `Notes`, and `Examples` sections as needed. Include math with `.. math::` for Sphinx rendering.

```python
def foo(x: jax.Array, scale: float = 1.0) -> jax.Array:
    """One-line summary of what the function does.

    Optional 1-2 sentence extended description.

    Parameters
    ----------
    x : jax.Array, shape (N, 3)
        Description of x.
    scale : float
        Description of scale (default 1.0).

    Returns
    -------
    jax.Array, shape (N, 3)
        Description of return value.

    Raises
    ------
    ValueError
        When x has the wrong shape.
    """
```

Private helpers (names starting with `_`) only need a one-line summary.

#### What to keep out of docstrings

- No descriptions of prior implementations ("This replaces the former …").
- No internal design rationale ("Why this class exists", "Design notes", "Path C", numbered pipeline steps). Move these to inline comments if they are genuinely needed by a maintainer reading the code.
- No architecture "brags". The docstring goal is to help the **user** understand the API, not to explain how clever the implementation is.

#### Section headers in code

All inline section-divider comments use the `# ===` style:

```python
# ============================================================================
# Section Name
# ============================================================================
```

Do **not** use `# ── Title ──────`, `# --- Title ---`, `# ---- # Title # ----`, or `# ---\n# Title\n# ---` variants.

### Module Scope

- `__init__.py` re-exports `CoilFEM`, `biot_savart`, `B_self_quadrature`, `lorentz_body_force`. Other modules are imported by explicit submodule path (e.g. `from coil_fem.meshing import rectangle_sweep`, `from coil_fem.geo import CurveXYZFourierJAX`).
- simsopt interop lives in `coil_fem.simsopt` — keep pure-JAX code simsopt-free where possible.

### Static vs. traced container convention

Two kinds of data bundles appear in this codebase; use the correct container for each.

**Traced bundles** (vary per optimisation step, flow through JAX autodiff):
- Use plain `dict` or `NamedTuple`.  Both are JAX pytrees.
- Example: `geom` dict returned by `SupportBeams.geometry(curves_jax, support_dofs)` contains endpoint positions, lengths, and DCMs — all traced arrays.
- Example: `support_dofs` passed to solvers and metrics.

**Static bundles** (fixed at construction, never traced):
- Use `@dataclasses.dataclass(frozen=True, eq=False)`.  The `eq=False` flag prevents JAX from treating the dataclass as a pytree leaf during hashing; the `frozen=True` flag enforces immutability.
- Example: `MonolithicStatic` in `coupling/drivers.py` — holds CSR patterns, cuDSS solver handles, and the pre-built `merged_solve` callable.
- **Never store traced JAX arrays on `self`.**  Traced values must always be passed as arguments so that JAX's tracing and autodiff machinery can see them.
- `CoilFEM.build_monolithic_static(solver)` is the canonical construction entry point for the monolithic static bundle; it is called once at `__init__` when `coupling == 'monolithic'` and `support.is_coupled`.

## Build and Packaging

- **Build system:** Hatchling (`pyproject.toml`).
- **Wheel contents:** `src/coil_fem` only.
- **No CI/CD** workflows are configured in-repo.
- **Sphinx docs:** Build with `make html` from `docs/`. API stubs in `docs/api/generated/` are gitignored; `autosummary_generate = True` in `docs/conf.py` recreates them on each build (required for Read the Docs).
- **Read the Docs:** Import the Git repository at [readthedocs.org](https://readthedocs.org), point the **configuration file** to `.readthedocs.yaml` at the repo root, and use the default **Sphinx** documentation type. The config installs `pip install -e ".[docs]"` and builds HTML from `docs/conf.py`. After the first successful build, set the project **canonical URL** in the RTD admin panel if you use a custom domain.

## Working with the Codebase

- Before editing a module, read it to understand existing patterns and public API.
- Tests should go in `tests/` following `test_<module>.py` naming.
- Do not commit data files (covered by `.gitignore`: `data/`, `*.npy`, `*.npz`, `*.h5`).
- Do not commit Jupyter checkpoints or build artifacts.
- When adding new modules, consider whether they need an optional-dependency guard.

## Support Structure Architecture

The coupling between coil FEM and support structures is split across three layers.

### `Support` ABC (`coupling/supports.py`)

`Support` is the abstract base class all support models must implement:

| Method | Required | Description |
|--------|----------|-------------|
| `is_coupled` | property | `True` when the support has its own DOFs |
| `solve(inputs)` | abstract | Advance support state; returns dict with `'u_s'` |
| `displacement_at(state, points)` | abstract | Support displacement at query points |
| `compute_weights(coil_idx, surf_pts, curves_jax, dofs)` | default=1 | Per-surface-node Winkler weights; `curves_jax` is the full list of all base-coil curves |
| `compute_attach(coil_idx, surf_pts, curves_jax, dofs, state)` | default=0 | Beam attachment displacement at surface nodes; same full-list signature |
| `coupling_terms(bcd, sdofs, surf_pts, coil_offsets, s_offset, surf_idx)` | default=empty | COO triplets for off-diagonal K_cs / K_sc blocks |
| `coo(bcd, sdofs, surf_pts)` | default stub | Support stiffness K_ss in COO format |
| `n_support_dofs` | attribute | Required when `is_coupled=True` |
| `k_lin` | attribute | Required when `is_coupled=True` (must match `winkler_k`) |

`Support` is the built-in uncoupled (grounded) support (`is_coupled=False`): it holds attachment points at zero displacement through a Winkler spring field whose spatial distribution is controlled by an optional `fixed_clamp_fn` callable.

### `SupportBeams` (`coupling/beam_networks.py`)

`SupportBeams` extends `Support` with a bisymmetric beam-network model.  It overrides `compute_weights`, `compute_attach`, `coupling_terms`, `coo`, and `solve`.

Key constructor arguments (all static; set once at construction):

- `n_beam_cc`, `n_beam_cf` — beam counts. CC beams have one entry per CC *group*: `n_base + 1` when `stellsym=True` (the extra last entry is the coil-0 `phi = 0` wrap group), else `n_base`. CF beams have one entry per base coil.
- `E`, `nu` — Young's modulus and Poisson's ratio.
- `cross_section_fn(support_dofs) -> (A, Iy, Iz, J)` — cross-section properties.
- `clamp_fn(surface_pts_beam_frame, dofs, sign_x) -> weights` — selects coil surface nodes for coupling.
  - `surface_pts_beam_frame`: `(n_surf, 3)` — surface points in the beam's local frame, origin at the endpoint, computed as `(pts − x_endpoint) @ Gamma_3` (column 0 is the true beam tangent).
  - `sign_x`: `True` at the node-1 end (beam extends toward `+x_local`), `False` at node-2.
- `k_lin`, `k_tor` — translational and torsional spring stiffness for endpoint-to-mesh coupling.

Optimisable quantities live in `support_dofs` (passed at solve time, never stored):

- `phis_start_cc`, `phis_end_cc` — attachment angles for CC beams: per-group lists, entry `g` of shape `(n_beam_cc[g],)` (`n_base + 1` entries when `stellsym=True`, else `n_base`).
- `phis_start_cf` — attachment angles for CF beams: per-coil list, entry `i` of shape `(n_beam_cf[i],)`.
- `x_foundation` — foundation anchor positions for CF beams: per-coil list, entry `i` of shape `(n_beam_cf[i], 3)`.
- `thetas_orientation_cc`, `thetas_orientation_cf` — cross-section roll angle per beam (same per-group / per-coil list layout as the attachment angles).

### Solver drivers (`coupling/drivers.py`)

Two module-level driver functions replace the uncoupled per-coil loop in `CoilFEM` when `support.is_coupled=True`:

- **`solve_staggered`** — Block Gauss-Seidel with Aitken relaxation.  Works on all backends (CPU and GPU).  Gradients are computed via a `@jax.custom_vjp` that applies the implicit-function theorem (IFT): the GMRES solve of `(I − dT/du_s)ᵀ λ = g` provides the correct adjoint without differentiating through the iteration history.  *Note:* the Python-loop forward pass is concrete (not JIT-compiled); wrapping the caller with `jax.jit` will fail.

- **`solve_monolithic`** — Assembles a single merged block matrix `[K_cc | K_cs; K_sc | K_ss]` and solves it with cuDSS in one shot.  Raises `NotImplementedError` when `solver != 'cudss'`.

### `CoilFEM` dispatch

`CoilFEM.__init__` accepts a `coupling='staggered'|'monolithic'` keyword (default `'staggered'`).  The internal `_solve_all` helper:

1. Builds per-coil mesh points, body forces, and Winkler weights.
2. Dispatches to `solve_staggered` or `solve_monolithic` when `support.is_coupled=True`.
3. Falls back to an independent per-coil loop when `support.is_coupled=False`.

When `is_coupled=True`, `CoilFEM` enforces `problem_options['winkler_k'] == support.k_lin` at construction.

### simsopt interop (`simsopt/optimizables.py`)

`CoilSupport` is the simsopt `Optimizable` base class that holds `base_coils`, `nfp`, `stellsym`, and the `Support` instance.  `CoilSupportFixed` and `CoilSupportTopBottom` are concrete subclasses.  `CoilFEMObjective` takes a single `CoilSupport` as its primary argument.

### Adding a new `Support` subclass

1. Subclass `Support` (to inherit the `fixed_clamp_fn` weight logic).
2. Set `is_coupled = True` (property) and declare `n_support_dofs` and `k_lin`.
3. Implement `solve(inputs) -> {'u_s': ...}` using `lineax` or any JAX-compatible solver.
4. Override `compute_weights` to return per-surface-node Winkler weights.
5. Override `compute_attach` to return the beam displacement at coil surface nodes (used by the staggered driver to shift the Winkler spring attachment point).
6. Implement `coupling_terms` to return the COO triplets for the off-diagonal K_cs / K_sc blocks (used by the monolithic driver).
7. Implement `coo` to return the support-local K_ss block in COO format (used by the monolithic driver).
8. Add tests in `tests/test_<subclass>.py`.
