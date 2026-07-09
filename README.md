# coil-fem

A differentiable FEA toolkit for stellarator coils.

## Installation

```bash
pip install -e --no-build-isolation ".[cudss]"     # (Recommended) Install with cuDSS, extra steps needed. See below.
pip install -e ".[dev]"                            # Editable install with test/notebook extras
pip install -e ".[docs]"                           # Sphinx documentation extras
```

`coil-fem` depends on `simsopt` and `jax`. For install instructions, 
please see:
- The [simsopt documentation](https://simsopt.readthedocs.io/latest/installation.html).
- The [JAX documentation](https://docs.jax.dev/en/latest/installation.html).


## Installing solvers

`coil-fem` supports multiple sparse solver that can be selected via
`problem_options`, e.g. `problem_options={'solver': 'umfpack'}` (and
`'adjoint_solver'` for the gradient pass). Currently supported solvers
are:

| Solver     | Type                                     | Availability                    |
|------------|------------------------------------------|-------------------------------- |
| `cudss`    | cuDSS direct sparse solver (recommended) | Installable extra, see below    |
| `umfpack`  | CPU sparse direct                        | Shipped with Scipy              |
| `petsc`    | PETSc (CPU/GPU)                          | Requires `petsc` and `petsc4py` |
| `jax`      | Pure-JAX iterative                       | Shipped with JAX                |
| `amgx`     | NVIDIA AmgX (GPU)                        | Requires `pyamgx`               |

- **`cudss` (recommended, extra setup needed)**
  The recommended solver for `coil-fem` is a GPU sparse direct solver based on
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


- **`umfpack`, `petsc`, `jax`, `amgx` (JAX-FEM built-in)** In addition to `cudss`,
  `coil-fem` supports a number of sparse linear solvers through JAX-FEM. For their
  setup instructions, please see the [JAX-FEM documentation](https://github.com/deepmodeling/jax-fem). 
  Note that *these solvers do not work with JAX JIT compilation*. As a result, their
  performance are heavily limited by CPU bottlenecks and/or array copying.
  