# Coil FEM Workflow Examples

This directory contains comprehensive examples demonstrating the complete coil-fem workflow.

## Main Workflow Notebook

**`docs/tutorial/coil_fem_workflow.ipynb`** - Complete workflow from geometry to FEM simulation (moved from `examples/` for documentation integration)

### What it covers:

1. **Curve Creation** - Define coil centerline using Fourier representation
2. **Frame Addition** - Add reference frame (normal/binormal vectors) using pure JAX
3. **Mesh Generation** - Create 3D tetrahedral mesh with automatic grid sizing
4. **Boundary Conditions** - Define top/bottom clamps based on Z-coordinate
5. **Load Application** - Apply body forces (gravity)
6. **FEM Setup** - Outline FEM system assembly (integrate with JAX-FEM)
7. **Visualization** - Plot displacement and stress fields
8. **Export** - Save mesh to VTU format for ParaView

### Key Features Demonstrated

✅ **Pure JAX Workflow**
- No simsopt.Curve dependency
- Fully differentiable pipeline
- Works with jax.jit, jax.grad, jax.vmap

✅ **Automatic Grid Sizing**
- Just specify `aspect_ratio`
- Grid resolution computed automatically
- Ensures mesh quality matches curve resolution

✅ **Simplified API**
- Direct use of FramedCurve objects
- Fewer parameters
- Clearer intent

✅ **Realistic Boundary Conditions**
- Top and bottom clamps
- Based on geometric criteria
- Visualized for verification

## Other Examples

### `framed_curve_jax_example.py`
Comprehensive examples of pure JAX framed curves:
- Basic usage
- RMF vs Centroid comparison
- Frame interpolation
- Frame rotation
- JAX transformations (jit, grad, vmap)

### `pure_jax_meshing_example.py`
End-to-end pure JAX meshing workflow:
- Rectangle and disk sweep
- Differentiation through meshing
- JIT compilation
- Batch processing with vmap
- Simple optimization loop

## Running the Notebook

### Prerequisites

```bash
pip install jax jaxlib numpy matplotlib
pip install coil-fem  # or install from source
```

### Optional (for full FEM):

```bash
pip install jax-fem meshio
```

### Launch Jupyter

```bash
jupyter notebook ../docs/tutorial/coil_fem_workflow.ipynb
```

Or use JupyterLab:

```bash
jupyter lab ../docs/tutorial/coil_fem_workflow.ipynb
```

## Workflow Overview

```
┌─────────────────┐
│  Curve DOFs     │  Fourier coefficients
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CurveXYZFourier │  Curve geometry
│      JAX        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FramedCurve     │  Add reference frame
│      JAX        │  (centroid or RMF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ rectangle_sweep │  Generate 3D mesh
│  or disk_sweep  │  (automatic sizing)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FramedCurveMesh      │  Tetrahedral mesh
│ (points, cells) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Boundary Conds  │  Top/bottom clamps
│   + Loads       │  Body forces
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FEM Solver     │  JAX-FEM or similar
│ (K*u = f)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Results      │  Displacement, stress
│ (visualization) │  Export to ParaView
└─────────────────┘
```

## Customization

### Change Coil Geometry

Modify the Fourier DOFs in cell 2:

```python
# Example: Simple circle
dofs = dofs.at[2].set(1.0)   # x: cos(2*pi*phi)
dofs = dofs.at[8].set(1.0)   # y: sin(2*pi*phi)

# Example: Helix
dofs = dofs.at[2].set(1.0)   # x: cos(2*pi*phi)
dofs = dofs.at[8].set(1.0)   # y: sin(2*pi*phi)
dofs = dofs.at[14].set(0.5)  # z: linear rise
```

### Change Cross-Section

```python
# Rectangular
mesh = rectangle_sweep(
    framed_curve,
    w_1=0.03,  # Wider
    w_2=0.01,  # Narrower
    aspect_ratio=1.0
)

# Circular
mesh = disk_sweep(
    framed_curve,
    radius=0.02,
    aspect_ratio=1.0
)
```

### Change Mesh Resolution

```python
# Finer mesh (smaller elements)
mesh = rectangle_sweep(
    framed_curve,
    w_1=0.02, w_2=0.02,
    aspect_ratio=0.5  # Finer cross-section
)

# Coarser mesh (larger elements)
mesh = rectangle_sweep(
    framed_curve,
    w_1=0.02, w_2=0.02,
    aspect_ratio=2.0  # Coarser cross-section
)

# Explicit control
mesh = rectangle_sweep(
    framed_curve,
    w_1=0.02, w_2=0.02,
    n_grid_1=10,  # Explicit
    n_grid_2=10   # Explicit
)
```

### Change Boundary Conditions

```python
# Larger clamp regions
clamp_thickness = 0.10 * z_range  # 10% instead of 5%

# Different criteria (e.g., based on radius)
r = jnp.sqrt(mesh.points[:, 0]**2 + mesh.points[:, 1]**2)
outer_nodes = jnp.where(r > 1.2)[0]  # Clamp outer nodes
```

### Change Material Properties

```python
# Steel instead of copper
E = 200e9   # Young's modulus (Pa)
nu = 0.30   # Poisson's ratio
rho = 7850  # Density (kg/m^3)

# Aluminum
E = 69e9
nu = 0.33
rho = 2700
```

## Integration with JAX-FEM

For full FEM solution, integrate with JAX-FEM:

```python
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh

# Create JAX-FEM mesh
jax_mesh = Mesh(mesh.points, mesh.cells)

# Define problem
problem = Problem(jax_mesh, vec=3)  # 3D vector problem
problem.set_params({'E': E, 'nu': nu})

# Boundary conditions
def dirichlet_bc(point):
    z = point[2]
    return (z < z_min + clamp_thickness) or (z > z_max - clamp_thickness)

problem.add_Dirichlet_bc(dirichlet_bc, 0, [0., 0., 0.])

# Body force
problem.add_body_force(body_force)

# Solve
sol = solver(problem)
displacement = sol[0]
```

## Visualization with ParaView

1. Run the notebook to generate `coil_mesh.vtu`
2. Open ParaView
3. File → Open → `coil_mesh.vtu`
4. Click "Apply" in Properties panel
5. Add filters:
   - **Slice**: Cut through mesh
   - **Clip**: Remove parts
   - **Warp By Vector**: Show displacement
   - **Glyph**: Show vectors

## Troubleshooting

### "No module named 'jax'"
```bash
pip install jax jaxlib
```

### "No module named 'coil_fem'"
```bash
cd /path/to/coil-fem
pip install -e .
```

### "Mesh generation is slow"
- Reduce `n_phi` (fewer quadrature points)
- Increase `aspect_ratio` (coarser mesh)
- Use explicit grid sizes for control

### "Out of memory"
- Reduce mesh resolution
- Use `mesh_type="TET4"` instead of `"TET10"`
- Process on GPU if available

## References

- **Coilforce Documentation**: See `../MESHING_API_EXAMPLES.md`
- **JAX-FEM**: https://github.com/tianjuxue/jax-fem
- **ParaView**: https://www.paraview.org/
- **JAX**: https://github.com/google/jax

## Support

For issues or questions:
1. Check documentation in parent directory
2. Review other examples in this directory
3. See `FINAL_SUMMARY.md` for complete API overview
