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
  container.py                 # CoilFEM — differentiable FEM pipeline container
  magnetic.py                  # B-field helpers (biot_savart, B_self_quadrature)
  forces.py                    # Lorentz body-force density
  elasticity.py                # LinearElasticity3D — JAX-FEM Problem subclass
  thermal.py                   # Thermal eigenstrain hooks (itc_strain, cauchy_stress_with_thermal_strain)
  metrics.py                   # Von Mises / strain metrics on FEM solutions
  meshing.py                   # Fixed-topology hex/tet meshing (rectangle/disk sweep, curved-sided TET10)
  problem.py                   # DeviceProblem — JAX device-assembly Problem subclass
  geo/                         # Curve geometry and symmetry subpackage
    __init__.py                # re-exports CurveXYZFourierJAX, framed curves, symmetry helpers
    curve_jax.py               # CurveXYZFourierJAX — JAX pytree, simsopt interop
    framed_curve_jax.py        # FramedCurveCentroidJAX / FramedCurveRMFJAX
    symmetries.py              # Stellarator symmetry expansion (pure JAX)
  simsopt/                     # simsopt Optimizable interop subpackage
    __init__.py                # re-exports CoilFEMObjective, CoilSupport family
    objective.py               # CoilFEMObjective — simsopt Optimizable wrapper
    support.py                 # CoilSupport, CoilSupportDiscrete, CoilSupportTopBottom
  backend/                     # Optional GPU backend subpackage
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

### Docstrings

Use NumPy-style docstrings with `Parameters`, `Returns`, and `Examples` sections. Include math with `.. math::` for Sphinx rendering.

### Module Scope

- `__init__.py` re-exports `CoilFEM`, `biot_savart`, `B_self_quadrature`, `lorentz_body_force`. Other modules are imported by explicit submodule path (e.g. `from coil_fem.meshing import rectangle_sweep`, `from coil_fem.geo import CurveXYZFourierJAX`).
- simsopt interop lives in `coil_fem.simsopt` — keep pure-JAX code simsopt-free where possible.

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
