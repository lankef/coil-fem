# Clearing simsopt's global JAX CPU pin

Design note for `coil_fem.gpu_env`.

## Problem

Importing simsopt mutates global JAX state: `simsopt/geo/jit.py` calls
`jax.config.update("jax_platform_name", "cpu")` at import time, pinning JAX's
default device to the host for the whole process.

`coil_fem` is GPU-first and hard-depends on simsopt, so that pin must be cleared
or every GPU install silently runs on the CPU.

## Symptom that exposed it

```
RuntimeError: Unknown backend cpu. Available backends are ['cuda']
  File ".../coil_fem/simsopt/coil_support_beams.py", line 298, in __init__
    estimated_R = jnp.sqrt(centers_zero[0]**2 + centers_zero[1]**2)
```

Triggered by setting `JAX_PLATFORMS=cuda` in a job script.

## Root cause

The previous fix lived in `magnetic.py` and was guarded:

```python
if os.environ.get("JAX_PLATFORMS") is None:
    jax.config.update("jax_platform_name", None)
```

Stated rationale: *"overwriting a deliberate `JAX_PLATFORMS=cpu` setting would
silently break GPU-free installs."*

That rationale does not hold. `jax.config.update("jax_platform_name", None)`
restores auto-selection; on a machine with no CUDA backend registered,
auto-selection picks CPU. Nothing breaks. The guard defends against a failure
mode that does not exist, while creating two real ones:

| `JAX_PLATFORMS` | Guard fires? | Result with the guard |
| --- | --- | --- |
| unset | repairs | correct (GPU) |
| `cpu` | skips | correct (CPU) — but correct without the guard too |
| `cuda` | skips | **hard crash** — pin requests an unregistered `cpu` backend |
| `cuda,cpu` | skips | **silently runs on CPU** — pin overrides the priority order |

The last row is the more dangerous of the two: no error, just a GPU-first FEM
workload quietly running on the host.

The two mechanisms do different jobs. `JAX_PLATFORMS` controls which backends
are *registered* and their priority order; `jax_platform_name` sets a
default-device preference. When they contradict each other, a leftover
third-party pin should never be the thing that wins.

## Fix

Clear the pin unconditionally, in a dedicated module, called once from the
bottom of the package `__init__`.

### 1. Additions to file: `src/coil_fem/gpu_env.py`

Defines `clear_simsopt_cpu_pin()`. Unconditional with respect to
`JAX_PLATFORMS`. Idempotent. Warns (does not fail) if called after the JAX
backend is already live, since arrays committed before that point cannot be
retroactively moved.

The module only *defines* the function — it does not self-apply on import.

### 2. `src/coil_fem/__init__.py`

```python
from .gpu_env import clear_simsopt_cpu_pin
from .coil_fem import CoilFEM

# MUST stay last: the import above pulls in coil_fem.magnetic, which imports
# simsopt at module level and thereby applies simsopt's global JAX CPU pin.
# Clearing it here -- after that import, before any user code -- is the whole
# fix. Do not move this above `from .coil_fem import CoilFEM`.
clear_simsopt_cpu_pin()

__all__ = ["CoilFEM"]
```

### 3. `src/coil_fem/magnetic.py`

Delete the `import os` and the guarded `jax.config.update` block (lines ~20–37).
Replace with a pointer comment so it is not reintroduced:

```python
# NOTE: simsopt's process-wide jax_platform_name="cpu" pin is cleared in
# coil_fem.gpu_env, invoked from coil_fem/__init__.py.
from simsopt.field.selffield import B_regularized_pure
```

## Why the call position is load-bearing

Clearing the pin only works if simsopt has *already* been imported — otherwise
it is a no-op that simsopt immediately undoes. Two facts make the bottom of
`__init__.py` the right place:

**`magnetic.py:27` is the sole eager simsopt import in the core package.**
Every other one is deferred:

```
src/coil_fem/magnetic.py:27               from simsopt.field.selffield import ...   <- module level
src/coil_fem/geo/curve_jax.py:77          inside to_simsopt()                        <- deferred
src/coil_fem/coil_fem.py:1545             inside a plotting branch                   <- deferred
src/coil_fem/simsopt/coil_support.py:154  try: ... (only on `import coil_fem.simsopt`)
src/coil_fem/simsopt/objectives.py:20-21  try: ... (only on `import coil_fem.simsopt`)
```

`magnetic` is reached via `from .coil_fem import CoilFEM` (`coil_fem.py:33`), so
simsopt is fully imported before the last line of `__init__` runs.

**Nothing initialises the JAX backend during import.** The only module-level JAX
statements in the package are a `jax.tree_util.register_pytree_node` call and a
`jax.jit(jax.vmap(...))` definition. Neither touches a device — `jit` is lazy and
nothing traces until first call. So there is no window in which a device is
chosen while the pin is still active, and no separate top-of-`__init__` call is
needed.

This is why an earlier draft's `force_simsopt_import=True` machinery (an
`import simsopt.geo` inside `gpu_env` purely for its side effect) was
dropped: the call position already provides the guarantee.

Note also that removing the `from .magnetic import ...` re-export from
`__init__.py` — a separate API-surface cleanup — does **not** affect any of
this. `coil_fem.py:33` imports `magnetic` regardless, so the import graph is
unchanged.

## Assumptions

1. `simsopt/geo/jit.py` is the only module that sets `jax_platform_name`.
   Verify: `grep -rn "jax_platform_name" ../simsopt/src`
2. At least one eager simsopt import is reached before the bottom of
   `coil_fem/__init__.py`. Currently `magnetic.py:27`; the specific line does
   not matter, only that one exists.

Assumption 2 would break silently if every simsopt import in the package were
made lazy. The test below asserts on end state rather than mechanism, so it
catches that.

## Test

Add to `tests/test_solver_claims.py`, which already asserts on the
`JAX_PLATFORMS` hint text. Must be a subprocess — the behaviour is import-time
and cannot be re-run in-process.

```python
@pytest.mark.parametrize("platforms", [None, "cpu", "cuda", "cuda,cpu"])
def test_simsopt_cpu_pin_is_cleared(platforms):
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    if platforms is not None:
        env["JAX_PLATFORMS"] = platforms
    code = ("import jax;"
            "from coil_fem.simsopt import CoilFEMObjective;"
            "from simsopt.mhd import Vmec;"
            "print(jax.config.jax_platform_name)")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "None"
```

Parametrising all four `JAX_PLATFORMS` values is the point: `"cuda"` is the case
that crashed, `"cuda,cpu"` is the silent-CPU case the old guard let through
undetected. The trailing `from simsopt.mhd import Vmec` checks a simsopt
subpackage that `coil_fem` does not itself import.

## Known limitation

If a user imports a simsopt subpackage that `coil_fem` never imports, *after*
importing `coil_fem`, simsopt re-applies the pin and nothing catches it. Closing
this properly would require an import hook — far more machinery than warranted.

Mitigations: `clear_simsopt_cpu_pin()` is exported so it can be called again,
and the late-call `RuntimeWarning` fires if the backend is already live. If the
`grep` in Assumption 1 shows multiple simsopt modules pinning independently, the
honest response is to document `clear_simsopt_cpu_pin()` as a public escape
hatch rather than implying the package fully contains the problem.

## Out of scope: GPU memory policy

`XLA_PYTHON_CLIENT_PREALLOCATE` and friends are deliberately **not** set here.

The distinction: repairing another library's global mutation is a legitimate
import-time side effect — leaving it broken is not an option and the correct
value is unambiguous. Choosing a memory policy is not — there is a defensible
default either way, and it belongs to the deployment (job script, or an explicit
opt-in helper such as `coil_fem.gpu_env.configure_gpu_memory()`).

Practically relevant for the cuDSS path: XLA preallocates 75% of the device by
default, and cuDSS allocates its factorisation *outside* XLA's pool with no
ability to borrow from it. Recommended in job scripts:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
# and do NOT set JAX_PLATFORMS -- it is redundant once this fix is in place
```