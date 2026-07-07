# coil-fem

A differentiable FEA toolkit for stellarator coils.

## Installation

```bash
pip install -e ".[dev]"       # editable install with test/notebook extras
pip install -e ".[docs]"      # Sphinx documentation extras
pip install -e ".[cudss]"     # cuDSS solver extras, see below.
```

`coil-fem` depends on `simsopt` and `jax`. For install instructions, 
please see:
- The [simsopt documentation](https://simsopt.readthedocs.io/latest/installation.html).
- The [JAX documentation](https://docs.jax.dev/en/latest/installation.html).


## Supported solvers

`coil-fem` supports multiple sparse solver that can be selected via
`problem_options`, e.g. `problem_options={'solver': 'umfpack'}` (and
`'adjoint_solver'` for the gradient pass). Currently supported solvers
are:

| Solver     | Type                                     | Availability         |
|------------|------------------------------------------|--------------------- |
| `umfpack`  | CPU sparse direct (default)              | Shipped with JAX-FEM |
| `petsc`    | PETSc (CPU/GPU)                          | Shipped with JAX-FEM |
| `jax`      | Pure-JAX iterative                       | Shipped with JAX-FEM |
| `amgx`     | NVIDIA AmgX (GPU)                        | Shipped with JAX-FEM |
| `cudss`    | cuDSS direct sparse solver (recommended) | Installable extra    |

- **`umfpack`, `petsc`, `jax`, `amgx` (JAX-FEM built-in)** — `coil-fem` supports a
  number of sparse linear solvers shipped with JAX-FEM. For their setup instructions,
  please see the [JAX-FEM documentation](https://github.com/deepmodeling/jax-fem).
  Note that *these solvers do not work with JAX JIT compilation*. As a result,
  their performance can be heavily limited by CPU bottlenecks and/or array copying.
  
- **`cudss` (recommended, extra setup needed)** — a GPU sparse direct solver via
  [spineax](https://github.com/johnviljoen/spineax) + NVIDIA cuDSS. This is a
  zero-copy solver that directly works with JIT-compiled JAX program on GPU.
  To install, follow the steps below:

  ```bash
  # Conda nvcc matching the CUDA 12 runtime (the pip nvcc wheel is incomplete):
  conda install -c conda-forge cuda-nvcc=<version>

  # Build spineax against the installed jaxlib/XLA headers and install the extra:
  pip install --no-build-isolation -e ".[cudss]"
  ```

  Notes: `--no-build-isolation` is required (spineax compiles CUDA against the
  installed `jaxlib`/XLA headers), and `nvidia-cudss-cu12` is pinned `<0.8`
  because cuDSS 0.8 introduced a breaking API change that spineax does not yet
  support. 
