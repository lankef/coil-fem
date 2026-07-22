# Comparing coil-fem with COMSOL

Step-by-step instructions for reproducing a `CoilFEM` structural analysis in
COMSOL Multiphysics using the **exact same** mesh, material properties, body
load, and boundary conditions.

---

## 1. Export the coil-fem results

Run the forward solve and write the VTU files:

```python
fem = CoilFEM(
    base_curves     = base_curves,
    base_currents   = base_currents,
    nfp             = nfp,
    stellsym        = stellsym,
    mesh_options    = mesh_options,
    material_options = material_options,   # E, nu, density
    fixed_clamp_fn  = fixed_clamp_fn,      # Winkler BC (or None)
    support_dofs    = support_dofs,
    problem_options = problem_options,     # winkler_k
)

fem.save_run_vtu(out_dir="vtu_out", prefix="coil")
```

This writes `vtu_out/coil00_run.vtu` (one file per base coil) containing:

| Field | Type | Unit | Description |
|---|---|---|---|
| `displacement_m` | point, vector | m | Nodal displacement |
| `von_mises_MPa` | cell, scalar | MPa | Quad-averaged von Mises stress |
| `f_vol_Npm3` | cell, vector | N/m³ | Lorentz + gravity body-force density |
| `B_self_T` | cell, vector | T | Self-field (quad-averaged) |
| `B_ext_T` | cell, vector | T | Mutual (external) field (quad-averaged) |
| `support_weight` | point, scalar | — | Winkler weight ∈ [0, 1] (if `fixed_clamp_fn` set) |
| `spring_k_Npm3` | point, scalar | N/m³ | Effective spring stiffness = `winkler_k × support_weight` |

Print the parameters you will need in COMSOL:

```python
print("E        =", fem._E,   "Pa")
print("nu       =", fem._nu)
print("density  =", fem._rho, "kg/m3")
print("winkler_k=", fem.problem_options.get('winkler_k', 'N/A'), "N/m3")
print("gravity  =", fem.gravity_options)   # None if not set
```

---

## 2. Import the mesh into COMSOL

1. **File → Import → CAD / Mesh** (or *Mesh → Import*).
2. Select `coil00_run.vtu`.  COMSOL reads VTU natively (VTK unstructured grid).
   - Element type: **Tet 4** (linear tetrahedra) or **Tet 10** if you built the
     mesh with `mesh_type='TET10'`.
3. After import, COMSOL creates a geometry domain from the imported mesh.
   Right-click the imported mesh and choose **Form Union** so that boundary
   selections are available.

---

## 3. Add a Solid Mechanics physics

**Physics → Add → Structural Mechanics → Solid Mechanics (solid)**

Set the domain to the imported mesh domain.

---

## 4. Material properties

**Materials → Add Material → Blank Material**, domain = all.

| Property | COMSOL name | Value |
|---|---|---|
| Young's modulus | `E` | `fem._E` Pa |
| Poisson's ratio | `nu` | `fem._nu` |
| Density | `rho` | `fem._rho` kg/m³ |

*Do not enable plasticity, thermal expansion, or any other physics unless
you are comparing the corresponding coil-fem features.*

---
<!-- 
## 5. Body load (Lorentz force)

**Solid Mechanics → Body Load**, domain = all.

The body force vector per unit volume was exported as the cell field
`f_vol_Npm3`.  To apply it in COMSOL:

1. In ParaView (or any VTK viewer), verify the three components of
   `f_vol_Npm3` look physically reasonable.
2. In COMSOL, set the load type to **Volume force** and enter each component
   as an **Interpolation function** defined on the mesh:
   - **Functions → Interpolation** → source = *File*, import the VTU, pick
     component `f_vol_Npm3[0]`, `f_vol_Npm3[1]`, `f_vol_Npm3[2]`.
   - Alternatively, export the body-force values to a CSV with one row per
     cell centroid and import that as a spatially varying load.

> **Shortcut**: For a quick sanity check, read the quad-averaged force
> magnitude `f_vol_mag_Npm3` and apply a **uniform** load of that magnitude
> in the dominant direction.  This will not match exactly but lets you
> verify the order of magnitude quickly. -->

### Uniform-current model (analytical alternative)

coil-fem builds the body force as `f = J × B` where
`J = (I / A) * t_hat` at every quadrature point.  To replicate this
analytically in COMSOL without importing a field:

1. Add a **Current** physics (or simply compute it by hand):
   - Current `I` [A] = `float(fem.base_currents[0])`
   - Cross-section area `A` [m²] = `fem.meshes[0].cross_section_area`
2. For a first-principles COMSOL study with the **AC/DC Module**:
   - Add **Magnetic Fields (mf)** physics.
   - Define a **Single-Turn Coil Domain** with the imported mesh as the
     conductor domain, with current = `I`.  COMSOL will solve for `B` and
     compute the Lorentz force automatically.
   - Cross-check the COMSOL `B_norm` against `B_self_mag_T` +
     `B_ext_mag_T` from the VTU.

---

## 6. Boundary conditions

### 6a. Winkler elastic foundation (distributed spring)

If `fixed_clamp_fn` was supplied, the point field `spring_k_Npm3` in the VTU
gives the spatially varying Winkler stiffness (N/m³).

In COMSOL:

**Solid Mechanics → Spring Foundation**, boundary = outer surface.

| Setting | Value |
|---|---|
| Spring type | Distributed, per unit volume → **Foundation stiffness** |
| `kx = ky = kz` | `problem_options['winkler_k']` N/m³ multiplied by the spatial weight |

For the spatially varying case, import `spring_k_Npm3` as an interpolation
function and enter it as the stiffness (same workflow as the body load above).
For a binary clamp (weight = 0 or 1), identify the supported nodes from
`support_weight > 0.5` and apply a **Fixed Constraint** on those boundary
selections instead of a spring — this is equivalent in the limit
`winkler_k → ∞`.

### 6b. No support / Dirichlet BC

If `fixed_clamp_fn = None`, coil-fem enforces a Dirichlet condition on specific
boundary nodes through `support_dofs`.  Apply the same **Fixed Constraint** on
the corresponding boundary selection in COMSOL.

---

## 7. Solve and compare

Run **Study → Stationary** in COMSOL.

Compare the following quantities between coil-fem and COMSOL:

| Quantity | coil-fem field | COMSOL expression |
|---|---|---|
| Displacement magnitude | `displacement_m` | `solid.disp` |
| Von Mises stress | `von_mises_MPa` × 1e6 | `solid.mises` |
| Body-force magnitude | `f_vol_mag_Npm3` | `solid.bx^2+…` (from load) |

Expected agreement for the same mesh and linear-elastic constitutive law:
**better than 1 % relative difference in peak von Mises stress** (residual
comes from quadrature weight differences and the order in which COMSOL
integrates cell loads).

---

## 8. Checklist

- [ ] Same mesh (import the VTU directly into COMSOL — do not re-mesh).
- [ ] Same `E`, `nu`, `density`.
- [ ] Same body force: import `f_vol_Npm3` cell field, or re-derive analytically
      with the same `I` and cross-section area.
- [ ] Same BC: either Winkler spring with `winkler_k` and the spatial weight, or
      Fixed Constraint on the supported nodes.
- [ ] Gravity: enable if `fem.gravity_options` is not `None`, with
      `g_vec` = `fem.gravity_options.get('g_vec', [0, 0, -9.80665])` m/s².
- [ ] Linear-static study (no dynamics, no plasticity).
