# coil-fem

Coil body forces, gradients, and optional JAX-FEM structural analysis for
stellarator coils.

## Installation

```bash
pip install -e ".[dev]"      # editable install with test/notebook extras
pip install -e ".[docs]"     # Sphinx documentation extras
```

`simsopt` is installed from a local editable checkout at `../simsopt` and is not
declared in `pyproject.toml`. See [AGENTS.md](AGENTS.md) for full environment
setup.

## Supported solvers

The linear/Newton solver used by the FEM structural analysis is selected via
`problem_options`, e.g. `problem_options={'solver': 'umfpack'}` (and
`'adjoint_solver'` for the gradient pass). Recognised values:

| Solver     | Type                         | Availability        |
|------------|------------------------------|---------------------|
| `umfpack`  | CPU sparse direct (default)  | JAX-FEM built-in    |
| `petsc`    | PETSc (CPU/GPU)              | JAX-FEM built-in    |
| `jax`      | Pure-JAX iterative           | JAX-FEM built-in    |
| `amgx`     | NVIDIA AmgX (GPU)            | JAX-FEM built-in    |
| `cudss`    | GPU sparse direct (cuDSS)    | coil-fem extra      |

- **`umfpack`, `petsc`, `jax`, `amgx` (JAX-FEM built-in)** — these ship with
  JAX-FEM; no extra installation is required from coil-fem. Refer to the
  [JAX-FEM documentation](https://github.com/deepmodeling/jax-fem) for setup and
  any per-solver dependencies (e.g. PETSc, AmgX). MUMPS is available through the
  `petsc` option as PETSc's LU factor package — it is not a separate solver
  name.
- **`cudss` (coil-fem extra)** — a GPU sparse direct solver via
  [spineax](https://github.com/johnviljoen/spineax) + NVIDIA cuDSS, providing a
  zero-copy on-device Newton/adjoint path. It is an optional dependency:

  ```bash
  # A real nvcc matching the CUDA 12.9 runtime (the pip nvcc wheel ships only ptxas):
  conda install -c conda-forge cuda-nvcc=12.9.86

  # Build spineax against the installed jaxlib/XLA headers and install the extra:
  pip install --no-build-isolation -e ".[cudss]"
  ```

  Notes: `--no-build-isolation` is required (spineax compiles CUDA against the
  installed `jaxlib`/XLA headers), and `nvidia-cudss-cu12` is pinned `<0.8`
  because cuDSS 0.8 introduced a breaking API change that spineax does not yet
  support. See [AGENTS.md](AGENTS.md) for details.
