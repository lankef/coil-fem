# Comparing coil-fem with ANSYS (Student Edition)

Step-by-step tutorial for reproducing a `CoilFEM` structural analysis in ANSYS
Workbench / Maxwell 3D / Mechanical using the **exact same** meshes, material
properties, coilset, Biot-Savart-derived loads, and boundary conditions —
without reading `f_vol_Npm3` from the coil-fem VTU, so that ANSYS independently
re-derives the forces.

> **Student Edition limits (as of ANSYS 2025 R1/R2)**
> - Maxwell 3D: 64 000 volume elements, 8 000 surface elements, 2 000 triangles (2D).
> - Mechanical: no hard element-count limit, but advanced ACT features and HPC
>   licences are absent.  Static-Structural, Spring Foundation, and External Data
>   all work within the Student licence.
>
> Strategies for staying within Maxwell limits are called out throughout.

---

## Contents

1. [Export from coil-fem](#1-export-from-coil-fem)
2. [Convert VTU meshes for ANSYS](#2-convert-vtu-meshes-for-ansys)
3. [Reproduce the full coilset in Maxwell](#3-reproduce-the-full-coilset-in-maxwell)
4. [Maxwell 3D magnetostatic setup](#4-maxwell-3d-magnetostatic-setup)
5. [Export B fields from Maxwell at exact locations](#5-export-b-fields-from-maxwell-at-exact-locations)
6. [Mechanical structural setup](#6-mechanical-structural-setup)
7. [Apply the Winkler spring foundation](#7-apply-the-winkler-spring-foundation)
8. [Solve and compare](#8-solve-and-compare)
9. [Batch workflow over multiple mesh-density folders](#9-batch-workflow-over-multiple-mesh-density-folders)
10. [Checklist](#10-checklist)

---

## 1. Export from coil-fem

Run the forward solve and write all VTU files plus the parameter record you will
need to re-enter in ANSYS.

```python
from coil_fem import CoilFEM

fem = CoilFEM(
    base_curves      = base_curves,
    base_currents    = base_currents,
    nfp              = nfp,
    stellsym         = stellsym,
    mesh_options     = mesh_options,
    material_options = material_options,
    support_fn       = support_fn,
    support_dofs     = support_dofs,
    problem_options  = problem_options,
)

# Forward solve + export — one *_run.vtu per base coil
paths = fem.save_run_vtu(out_dir="vtu_out", prefix="coil")
```

This writes, for each base coil `i`:

```
vtu_out/coil{i:02d}_run.vtu
```

| Field | Type | Unit | Content |
|---|---|---|---|
| `displacement_m` | point, vector | m | Nodal displacement (n_nodes × 3) |
| `support_weights` | point, scalar | — | Winkler weight ∈ [0, 1] |
| `spring_k_Npm3` | point, scalar | N/m³ | `winkler_k × support_weight` |
| `von_mises_MPa` | cell, scalar | MPa | Quad-averaged von Mises stress |
| `f_vol_Npm3` | cell, vector | N/m³ | Lorentz + gravity body-force density |
| `B_self_T` | cell, vector | T | Self-field (quad-averaged over each cell) |
| `B_ext_T` | cell, vector | T | Mutual field from all other coils (quad-averaged) |

> **Note on quad-averaging.** Every cell field is the mean over all FEM
> quadrature points inside that cell — not a centroid interpolation.
> ANSYS Mechanical element-averaged results (output scoping set to
> *Element Mean* or APDL `ETABLE` averaged results) are the correct
> counterpart.

Record the key parameters you will re-enter in ANSYS:

```python
import numpy as np

print("=== Material ===")
print(f"E            = {fem._E:.6e}  Pa")
print(f"nu           = {fem._nu}")
print(f"density      = {fem._rho}  kg/m3")

print("\n=== Winkler BC ===")
print(f"winkler_k    = {fem.problem_options.get('winkler_k', 'N/A')}  N/m3")

print("\n=== Gravity ===")
print(f"gravity      = {fem.gravity_options}")   # None = off

print("\n=== Coil currents (base coils) ===")
for i, I in enumerate(fem.base_currents):
    A = fem.meshes[i].cross_section_area
    print(f"  coil {i:02d}: I = {float(I):.4e} A,  A = {A:.6e} m2,"
          f"  J = {float(I)/A:.4e} A/m2")

print("\n=== Symmetry ===")
print(f"nfp          = {fem.nfp}")
print(f"stellsym     = {fem.stellsym}")
print(f"n_base       = {len(fem.base_curves)}")
print(f"n_total      = {fem.n_total}   (= n_base * nfp * (1 + int(stellsym)))")

print("\n=== Mesh shapes ===")
for i, m in enumerate(fem.meshes):
    if m.shape == 'rect':
        print(f"  coil {i:02d}: rect  w1={m.w1:.4e} m  w2={m.w2:.4e} m")
    else:
        print(f"  coil {i:02d}: disk  radius={m.radius:.4e} m")
```

Also export the **centerline quadrature points** for every base coil and every
symmetry image — these are needed in Section 3 to place coil conductors in Maxwell:

```python
import numpy as np
from coil_fem.geo import (
    apply_symmetries_to_gammas,
    apply_symmetries_to_gammadashs,
    apply_symmetries_to_currents,
)
import jax.numpy as jnp

base_gammas    = jnp.stack([c.gamma()      for c in fem.base_curves])
base_gammadashs= jnp.stack([c.gammadash()  for c in fem.base_curves])

all_gammas     = apply_symmetries_to_gammas(base_gammas, fem.nfp, fem.stellsym)
all_gammadashs = apply_symmetries_to_gammadashs(base_gammadashs, fem.nfp, fem.stellsym)
all_currents   = apply_symmetries_to_currents(fem.base_currents, fem.nfp, fem.stellsym)

import os
os.makedirs("vtu_out", exist_ok=True)
for j in range(fem.n_total):
    np.savetxt(
        f"vtu_out/coil_all_{j:03d}_centerline.csv",
        np.asarray(all_gammas[j]),
        header="x,y,z", delimiter=",", comments="",
    )
np.savetxt("vtu_out/all_currents.csv",
           np.asarray(all_currents)[:, None],
           header="I_A", delimiter=",", comments="")

print("Centerline CSVs written for all", fem.n_total, "coils")
```

Finally, export **cell-center coordinates and spring stiffness** so you can map
the ANSYS results back onto the exact same cell locations:

```python
import meshio

for vtu_path in paths:
    m = meshio.read(vtu_path)
    pts   = m.points                              # (n_nodes, 3)
    cells = m.cells[0].data                      # (n_cells, 4) — TET4 connectivity
    # Cell centroid = mean of 4 corner nodes
    centroids = pts[cells].mean(axis=1)          # (n_cells, 3)
    base = vtu_path.replace("_run.vtu", "")
    np.savetxt(base + "_cell_centers.csv", centroids,
               header="x,y,z", delimiter=",", comments="")
    # Also save reference values for comparison
    vm  = m.cell_data["von_mises_MPa"][0]
    Bs  = m.cell_data["B_self_T"][0]
    Be  = m.cell_data["B_ext_T"][0]
    disp= m.point_data["displacement_m"]
    np.savetxt(base + "_ref_cell.csv",
               np.column_stack([centroids, vm,
                                Bs[:, 0], Bs[:, 1], Bs[:, 2],
                                Be[:, 0], Be[:, 1], Be[:, 2]]),
               header="x,y,z,vm_MPa,Bsx,Bsy,Bsz,Bex,Bey,Bez",
               delimiter=",", comments="")
    np.savetxt(base + "_ref_nodes.csv",
               np.column_stack([pts, disp]),
               header="x,y,z,ux,uy,uz", delimiter=",", comments="")
```

After this step your output folder looks like:

```
vtu_out/
  coil00_run.vtu
  coil00_cell_centers.csv      # reference cell centroids for ANSYS export
  coil00_ref_cell.csv          # reference B + von Mises values
  coil00_ref_nodes.csv         # reference nodal displacements
  coil_all_000_centerline.csv  # full coilset, coil index 0
  coil_all_001_centerline.csv
  ...
  all_currents.csv
```

---

## 2. Convert VTU meshes for ANSYS

ANSYS Mechanical does not import `.vtu` directly.  Convert each VTU to **Abaqus
Input** format (`.inp`), which Mechanical accepts through *External Model*:

```python
import meshio, glob

for vtu_path in sorted(glob.glob("vtu_out/coil*_run.vtu")):
    inp_path = vtu_path.replace("_run.vtu", ".inp")
    m = meshio.read(vtu_path)
    # meshio TET4 → Abaqus C3D4
    meshio.write(inp_path, m)
    print("wrote", inp_path)
```

> **TET4 vs TET10.**  coil-fem builds TET4 meshes by default
> (`mesh_options['mesh_type'] = 'TET4'`).  Abaqus C3D4 (= Ansys SOLID285 or
> SOLID72 equivalent) is correct.  Do not let Mechanical upgrade the elements
> to TET10 — this would change the mesh and invalidate the comparison.
> In Mechanical: *Mesh → Mesh Control → Element Order → Linear*.

In ANSYS Workbench:

1. **File → New** to create a new Workbench project.
2. Drag **External Model** from the *Component Systems* toolbox onto the
   *Project Schematic*.
3. Double-click *Setup* inside the External Model block.
4. Click **Add Input File**, browse to `coil00.inp`.
5. Click **Generate** and verify the mesh summary shows the expected number of
   nodes and elements (must match `n_nodes` and `n_cells` from the VTU).
6. Close.  The External Model block now supplies a mesh to downstream systems.

---

## 3. Reproduce the full coilset in Maxwell

This section explains which coils to model and how to place them.

### 3.1 Coilset expansion — the exact coil-fem convention

coil-fem expands `n_base` base coils into `n_total` coils using
`apply_symmetries_to_gammas` in `src/coil-fem/symmetries.py`.
The expansion order is:

```
n_total = n_base * nfp * (1 + int(stellsym))

for k in 0 .. nfp-1:
    for flip in [False] if not stellsym else [False, True]:
        for i in 0 .. n_base-1:
            # coil index = k*(1+int(stellsym))*n_base + flip_idx*n_base + i
            rotation_angle = 2*pi*k / nfp   (around z-axis)
            if flip:
                transform = diag(1, -1, -1)  (negate y and z, applied AFTER rotation)
                current   = -base_currents[i]
            else:
                transform = identity
                current   = +base_currents[i]
```

Key rules:
- **Rotation only** copies (flip=False, k>0): same `|current|` as base coil,
  positive sign.
- **Stellarator images** (flip=True): current sign is **negated**.
- The CSV files written in Section 1 encode all of this — coil `j` in
  `coil_all_{j:03d}_centerline.csv` carries the current in row `j` of
  `all_currents.csv`.

### 3.2 Building coil geometry in Maxwell

For each coil `j` (0 … n_total−1):

1. In Maxwell 3D, go to **Draw → Spline 3D** (or **Polyline 3D**).
2. Import the centerline points from `coil_all_{j:03d}_centerline.csv` by
   scripting (see below) or by pasting coordinates into the *Polyline Points*
   dialog.  The curve parameter runs 0 → 1 uniformly; close the loop by
   repeating the first point.
3. Assign a cross-section:
   - **Rectangular** coil (`shape=rect`): draw a rectangle of width `w1` ×
     height `w2` in the local frenet-serret normal/binormal plane.  In Maxwell,
     use **Draw → Sweep Along Spline** with that rectangle as profile.
   - **Disk** coil (`shape=disk`): draw a circle of radius `radius` and sweep.
4. Assign material **Copper** (or the actual conductor material) just for
   geometry purposes; the current density will be overridden in step 4.

For many coils this is best scripted through Maxwell's IronPython API:

```python
# Run inside Maxwell 3D scripting console
import ScriptEnv, csv, os
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop = ScriptEnv.GetDesktop()
oProject = oDesktop.GetActiveProject()
oDesign  = oProject.GetActiveDesign()
oEditor  = oDesign.GetActiveEditor()

coil_dir = r"C:\path\to\vtu_out"  # adjust

for j in range(n_total):  # set n_total from your print output above
    pts = []
    with open(os.path.join(coil_dir, f"coil_all_{j:03d}_centerline.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["x"]), float(row["y"]), float(row["z"])))
    pts.append(pts[0])  # close loop

    # Convert to the Maxwell polyline command list
    poly_pts = ["NAME:PolylinePoints"] + [
        f"  Name:='P{k}', X='{x}meter', Y='{y}meter', Z='{z}meter'"
        for k, (x, y, z) in enumerate(pts)
    ]
    # (abbreviated — consult Maxwell scripting guide for full syntax)
    # oEditor.CreatePolyline(["NAME:PolylineParameters", poly_pts, ...])
```

> **Student Edition and the 64 k element limit.**  The 64 000 volume element
> cap in Maxwell Student applies to the *Maxwell adaptive-mesh volume*.  For a
> magnetostatic solve with stranded conductors the EM mesh lives in air, not
> inside each conductor solid.  Use these tactics to stay under the limit:
>
> - Reduce the geometry bounding box to the smallest box enclosing just
>   the coil under study plus a few radii of clearance.
> - Use *Mesh Operations → Assign Surface Approximation* with a generous
>   normal deviation (e.g. 30°) to reduce surface element count.
> - Increase the convergence tolerance to 5–10 % if you are only validating
>   the order of magnitude of `B`.
> - As a last resort, run one base-coil region at a time but still model
>   **all other coils as external stranded sources** using Maxwell's
>   *Coil Terminal* / *Current Density* boundary to avoid meshing them as
>   volumes.

---

## 4. Maxwell 3D magnetostatic setup

### 4.1 Solution type

**Maxwell 3D → Solution Type → Magnetostatic.**

Magnetostatic is the correct choice: coil-fem uses a static Lorentz force
with no time variation or eddy currents.

### 4.2 Excitation: uniform current density

For each coil solid body `j`, assign a **Stranded** winding excitation:

| Setting | Value |
|---|---|
| Excitation type | *Stranded* |
| Number of conductors | 1 |
| Current | `all_currents[j]` A  (from `all_currents.csv`, row `j`) |
| Direction | Along the path tangent (Maxwell infers from geometry) |

> **Why Stranded, not Solid?**  coil-fem uses a uniform current-density model
> `J = (I / A) * t_hat` — every cross-section slice carries the same current
> flowing in the tangent direction with no skin-depth variation.  The Stranded
> winding type in Maxwell implements exactly this model.  Solid conductor would
> instead enforce current continuity with resistive diffusion, which is
> different.

### 4.3 Validate current direction vs. stellarator images

For stellarator-image coils (flip=True, index `j ≥ n_base` with the second
block in the expansion), the current **sign is negated**.  In the *Excitation*
dialog for those coils, enter `−|I|`.  You can read the sign directly from
`all_currents.csv` — rows with a minus sign are the stellarator images.

### 4.4 Boundary conditions

Assign a **Zero Tangential H** (natural Neumann) boundary on the outer box
surfaces.  Make the box large enough (≥ 3× the characteristic coil radius from
the origin) to keep the error from the truncated boundary below 1%.

### 4.5 Solve

Click **Analyze All**.  After convergence, verify:
- Reported current for each coil matches the intended value.
- The field plot of `|B|` near the structural coil is physically reasonable
  (order of `mu_0 * I / (2*pi*r)` for a straight wire of radius `r`).

---

## 5. Export B fields from Maxwell at exact locations

You need B at the **same cell centers** where coil-fem stored `B_self_T` and
`B_ext_T`.  Both quantities live in the cell-center CSV exported in Section 1.

### 5.1 B_total at cell centers

In the Maxwell **Fields Post-processor → Fields Calculator**, or via scripting:

```python
# Maxwell IronPython snippet (run after solve)
oModule = oDesign.GetModule("FieldsReporter")

# Export B at the locations from coil00_cell_centers.csv
oModule.ExportToFile(
    "B_total_coil00",            # named expression (define Bvec first if needed)
    r"C:\path\to\coil00_cell_centers.csv",   # input points file
    r"C:\path\to\coil00_ansys_B_total.csv",  # output file
    ["X", "Y", "Z", "Bx", "By", "Bz"],
    False,   # include points in output
    False,   # points in SI already
)
```

The input points file (`coil00_cell_centers.csv`) has the header `x,y,z`
already written by the Python script in Section 1.  Maxwell's `ExportToFile`
expects one point per row in metres.

### 5.2 Separating B_self from B_mutual

coil-fem stores `B_self_T` (field from the coil on itself) and `B_ext_T`
(field from all other coils) separately.  To reproduce this split in Maxwell:

| Quantity | ANSYS approach |
|---|---|
| `B_self_T` | Re-run Maxwell with **only coil `i`** active (zero all other currents), then export B. |
| `B_ext_T` (`B_mutual`) | Re-run with coil `i` current set to **zero**, all others active, export B. |
| `B_total` = `B_self + B_ext` | Single run with all coils active. |

The fastest path for comparison is to export `B_total` from the full solve and
compare it to `B_self_T + B_ext_T` element-wise from the VTU.

> **Tip — parametric solve.**  Define a Maxwell parametric sweep over an
> integer variable `active_coil` (0 … n_total−1) where each variation sets
> that coil's current to `I_j` and all others to 0.  Export B for each
> variation.  This lets you extract both `B_self` and `B_ext` in a single
> project run.

---

## 6. Mechanical structural setup

### 6.1 Link External Model to Static Structural

In Workbench:

1. Drag **Static Structural** from the toolbox onto the Project Schematic.
2. Drag the **Model** cell of the External Model block onto the **Model** cell
   of Static Structural.  Workbench will use the imported mesh exactly as-is.
3. In Mechanical: confirm that **Mesh → Element Order** shows *Linear*
   (TET4 / SOLID285).  Do not allow Mechanical to automatically refine or
   upgrade to quadratic.

### 6.2 Material

Add a **New Material**:

| Property | ANSYS field | Value |
|---|---|---|
| Young's modulus | Isotropic Elasticity → Young's Modulus | `fem._E` Pa |
| Poisson's ratio | Isotropic Elasticity → Poisson's Ratio | `fem._nu` |
| Density | Density | `fem._rho` kg/m³ |

Assign this material to all bodies.

### 6.3 Lorentz body load from Maxwell B field

The Lorentz force density that coil-fem applies is:

```
f_vol = J × (B_self + B_ext)  [N/m³]
```

where `J = (I / A) * t_hat` is the uniform current density.

**Preferred: direct Maxwell–Mechanical coupling in Workbench**

1. Ensure you have completed the Maxwell solve in the same Workbench project.
2. In the Project Schematic, connect the **Solution** cell of the Maxwell 3D
   design to the **Setup** cell of Static Structural.
3. Mechanical will automatically import the mapped Lorentz-force density from
   Maxwell onto every element of the structural mesh.
4. No manual force entry is needed.

**Fallback: External Data body force (if Workbench coupling is not available)**

If the two solvers are in separate projects, export the computed body force from
Maxwell's Fields Calculator and import it via **External Data** in Mechanical:

1. In Maxwell, use the Fields Calculator to compute `J × B` at the cell
   centroids of the structural mesh (using the same `coil00_cell_centers.csv`
   input file).  Export as a CSV with columns `x, y, z, fx, fy, fz`.
2. In Mechanical → **External Data** component:
   - Add a *File* entry, point to the CSV.
   - Map `fx/fy/fz` to a **Body Force Density** load (SI, N/m³).
3. Apply the External Data load to the body.

---

## 7. Apply the Winkler spring foundation

The coil-fem Winkler BC is implemented as a spatially weighted surface spring:
`k_eff(node) = winkler_k × support_weight(node)`.  The point field
`spring_k_Npm3` in the VTU already encodes the product `k_eff`.

### 7.1 Prepare the spring stiffness file

```python
import meshio, numpy as np

m = meshio.read("vtu_out/coil00_run.vtu")
pts = m.points
k   = m.point_data["spring_k_Npm3"]   # (n_nodes,) — zero on interior nodes

# Export only nodes where k > 0 (Winkler BC active)
mask = k > 0
np.savetxt(
    "vtu_out/coil00_spring_k.txt",
    np.column_stack([pts[mask], k[mask]]),
    header="x y z k_Npm3",
    comments="",
)
```

### 7.2 Import spring stiffness via External Data

In Workbench, add an **External Data** component and connect it to the Setup
cell of Static Structural:

1. Add the `coil00_spring_k.txt` file as an input.
2. Set column mapping:
   - Columns 1–3: X, Y, Z coordinates.
   - Column 4: **Spring Stiffness** (Elastic Foundation, N/m³).
3. Set *Weighting* to **Inverse Distance** and *Power* to 2 to interpolate
   stiffness values from node locations onto the mesh (since nodes are
   identical, the interpolation is exact; you may also set Power = 0 and use
   *Nearest Neighbour* for a hard nearest-point assignment).

### 7.3 Apply the Elastic Foundation in Mechanical

In Mechanical:

1. **Static Structural → Insert → Elastic Foundation**.
2. Scope: select the outer surface of the coil body (all surface faces).
3. Foundation Stiffness: set to **Imported** and link to the External Data
   quantity from step 7.2.

> **Uniform spring fallback.**  If the spring weights are binary (0 or 1,
> i.e. a simple clamp over one region), you can instead:
> - Use ParaView or the VTU Python export to identify which surface nodes have
>   `support_weights > 0.5`.
> - Export those node coordinates as a named selection file and apply a
>   **Fixed Constraint** in Mechanical on those face patches.

---

## 8. Solve and compare

Run **Analysis Settings → Solve** in Mechanical (Static Structural).

### 8.1 Export ANSYS results at the exact VTU locations

**Nodal displacement** — same points as coil-fem:

In Mechanical, insert a **Directional Deformation** (X, Y, Z) and a
**Total Deformation** result.  Set **Scoping → Mesh Node**.  ANSYS reports
displacements at every node; export via *Results → Export* to a CSV.

Then compare against `coil00_ref_nodes.csv` written in Section 1.

**Von Mises stress and B field at cell centers:**

coil-fem stores `von_mises_MPa` and `B` fields as **element (cell) averages**
over quadrature points.  To obtain the equivalent in ANSYS:

- In Mechanical: **Equivalent (von Mises) Stress → Scoping → All Elements**.
  Set *Result Averaging* to **None** (element values, not nodal-smoothed).
  Export.
- For B: Maxwell exports B at the cell-center coordinates you supplied; that
  matches the averaging convention directly.

After export, compare with the saved reference CSVs using Python:

```python
import numpy as np

ref  = np.loadtxt("vtu_out/coil00_ref_cell.csv", delimiter=",", skiprows=1)
ans  = np.loadtxt("coil00_ansys_vm.csv",          delimiter=",", skiprows=1)

# Align rows by closest cell-center (should be 1-to-1 if mesh is identical)
from scipy.spatial import cKDTree
tree = cKDTree(ref[:, :3])
_, idx = tree.query(ans[:, :3])

vm_ref  = ref[idx, 3]          # von_mises_MPa from coil-fem
vm_ans  = ans[:, 3]            # von_mises_MPa from ANSYS (in MPa)

rel_err = np.abs(vm_ans - vm_ref) / (np.abs(vm_ref) + 1e-10)
print(f"Max relative error von Mises: {rel_err.max()*100:.2f}%")
print(f"Mean relative error von Mises: {rel_err.mean()*100:.2f}%")
```

Expected agreement for identical mesh, material, load, and BC:
**< 1 % relative error in peak von Mises stress**.  Residual differences
come from quadrature-weight differences and floating-point averaging order.

### 8.2 Comparison table

| Quantity | coil-fem field | ANSYS equivalent |
|---|---|---|
| Nodal displacement | `displacement_m` (point, m) | Total/Directional Deformation (m) |
| Von Mises stress | `von_mises_MPa` (cell, MPa) | Equivalent Stress, Element Mean (MPa) |
| Self-field | `B_self_T` (cell, T) | Maxwell B with only this coil active, at cell centers |
| Mutual field | `B_ext_T` (cell, T) | Maxwell B with this coil zeroed, at cell centers |
| Total force density | `f_vol_Npm3` (cell, N/m³) | Maxwell–Mechanical coupled Lorentz force density |

---

## 9. Batch workflow over multiple mesh-density folders

If you have several folders of VTUs produced at different mesh densities, the
following layout and Python driver streamline repeating the full workflow.

### 9.1 Recommended directory layout

```
runs/
  coarse/          ← coil*_run.vtu from save_run_vtu(mesh_options={aspect_ratio:2})
  medium/          ← coil*_run.vtu from save_run_vtu(mesh_options={aspect_ratio:1})
  fine/            ← coil*_run.vtu from save_run_vtu(mesh_options={aspect_ratio:0.5})
```

Each subfolder will receive the derived files created by the Python steps above.

### 9.2 Batch preparation script

```python
"""batch_prep.py — run once to prepare all mesh-density folders."""
import glob, os, meshio, numpy as np

RUN_DIRS = sorted(glob.glob("runs/*/"))

for run_dir in RUN_DIRS:
    vtu_paths = sorted(glob.glob(os.path.join(run_dir, "coil*_run.vtu")))
    if not vtu_paths:
        print(f"[skip] {run_dir} — no VTU files")
        continue
    print(f"\n=== {run_dir} ===")

    for vtu_path in vtu_paths:
        stem = vtu_path.replace("_run.vtu", "")
        m    = meshio.read(vtu_path)
        pts  = m.points
        cells= m.cells[0].data

        # 1. Convert mesh to Abaqus .inp for Mechanical
        meshio.write(stem + ".inp", m)

        # 2. Cell centers
        centroids = pts[cells].mean(axis=1)
        np.savetxt(stem + "_cell_centers.csv", centroids,
                   header="x,y,z", delimiter=",", comments="")

        # 3. Reference result CSVs
        vm  = m.cell_data["von_mises_MPa"][0]
        Bs  = m.cell_data["B_self_T"][0]
        Be  = m.cell_data["B_ext_T"][0]
        disp= m.point_data["displacement_m"]
        np.savetxt(stem + "_ref_cell.csv",
                   np.column_stack([centroids, vm,
                                    Bs[:,0],Bs[:,1],Bs[:,2],
                                    Be[:,0],Be[:,1],Be[:,2]]),
                   header="x,y,z,vm_MPa,Bsx,Bsy,Bsz,Bex,Bey,Bez",
                   delimiter=",", comments="")
        np.savetxt(stem + "_ref_nodes.csv",
                   np.column_stack([pts, disp]),
                   header="x,y,z,ux,uy,uz", delimiter=",", comments="")

        # 4. Spring stiffness nodes
        if "spring_k_Npm3" in m.point_data:
            k = m.point_data["spring_k_Npm3"]
            mask = k > 0
            np.savetxt(stem + "_spring_k.txt",
                       np.column_stack([pts[mask], k[mask]]),
                       header="x y z k_Npm3", comments="")
        print(f"  prepared {os.path.basename(stem)}")

print("\nDone.  For each run_dir, create a separate ANSYS Workbench project.")
```

### 9.3 One Workbench project per mesh-density folder

Because the mesh changes between density levels, each folder requires its own
Workbench project (`.wbpj`).  The recommended approach for Student Edition
(which lacks scripted project creation) is:

1. Create a **template project** for one mesh density (e.g. `medium`).
2. For each other density, **duplicate** the `.wbpj` folder in Windows
   Explorer, then:
   - Open the copy in Workbench.
   - In *External Model → Setup*, replace the `.inp` path with the new
     density's `.inp`.
   - Click **Update Project** to re-import the mesh and re-solve.
3. After solving, export all result CSVs to the same density folder.

### 9.4 Batch comparison script

```python
"""batch_compare.py — run after all ANSYS projects are solved and exported."""
import glob, os, numpy as np
from scipy.spatial import cKDTree

RUN_DIRS = sorted(glob.glob("runs/*/"))

summary = []
for run_dir in RUN_DIRS:
    ref_paths = sorted(glob.glob(os.path.join(run_dir, "coil*_ref_cell.csv")))
    for ref_path in ref_paths:
        stem    = ref_path.replace("_ref_cell.csv", "")
        ans_vm  = stem + "_ansys_vm.csv"
        ans_b   = stem + "_ansys_B_total.csv"
        if not os.path.exists(ans_vm):
            print(f"[missing ANSYS file] {ans_vm}")
            continue

        ref = np.loadtxt(ref_path, delimiter=",", skiprows=1)
        ans = np.loadtxt(ans_vm,   delimiter=",", skiprows=1)

        tree = cKDTree(ref[:, :3])
        _, idx = tree.query(ans[:, :3])

        vm_ref = ref[idx, 3]
        vm_ans = ans[:, 3]
        rel    = np.abs(vm_ans - vm_ref) / (np.abs(vm_ref) + 1e-8)

        label = f"{run_dir.strip('/')}/{os.path.basename(stem)}"
        summary.append({
            "case":           label,
            "n_cells":        len(vm_ref),
            "vm_max_ref_MPa": vm_ref.max(),
            "vm_max_ans_MPa": vm_ans.max(),
            "rel_err_max_%":  rel.max() * 100,
            "rel_err_mean_%": rel.mean() * 100,
        })

print(f"\n{'Case':<40} {'N_cells':>8} {'vm_ref':>10} {'vm_ans':>10} "
      f"{'max_err%':>9} {'mean_err%':>10}")
print("-" * 95)
for s in summary:
    print(f"{s['case']:<40} {s['n_cells']:>8d} {s['vm_max_ref_MPa']:>10.2f} "
          f"{s['vm_max_ans_MPa']:>10.2f} {s['rel_err_max_%']:>9.2f} "
          f"{s['rel_err_mean_%']:>10.3f}")
```

---

## 10. Checklist

Work through this list before trusting the comparison.

### Mesh

- [ ] Mechanical uses the mesh imported from the VTU `.inp` — not a
  re-meshed geometry.
- [ ] Element order in Mechanical is **Linear** (TET4, no mid-side nodes).
- [ ] Node count and element count printed by Mechanical match those in
  the VTU (check `n_nodes = mesh.points.shape[0]`, `n_cells` from
  `fem.meshes[i].n_cells`).

### Material

- [ ] `E` = `fem._E` Pa.
- [ ] `nu` = `fem._nu`.
- [ ] `density` = `fem._rho` kg/m³.
- [ ] No plasticity, no temperature dependency (unless `material_options`
  contains `alpha` and temperature data).

### Coilset and current

- [ ] All `n_total` coils are modelled in Maxwell.  Coil count =
  `n_base * nfp * (1 + int(stellsym))`.
- [ ] Rotation angle for field-period `k` is `2π k / nfp` about the z-axis.
- [ ] Stellarator images (flip=True) use `diag(1, -1, -1)` applied after
  rotation.  Their current sign is **negated** relative to the base coil.
- [ ] Excitation type in Maxwell is **Stranded** (uniform current density).
- [ ] Current magnitude for each coil matches `abs(all_currents[j])`.
- [ ] Cross-section area used for `J = I/A` matches:
  - `A = w1 * w2` for rectangular cross-sections.
  - `A = π r²` for disk cross-sections.

### Boundary conditions

- [ ] Spring stiffness field `spring_k_Npm3` is imported unchanged (N/m³).
- [ ] Elastic Foundation is applied on the **outer surface only**, weighted by
  the spatially varying stiffness.
- [ ] If `support_fn = None` was used in coil-fem, apply a Fixed Constraint on
  the intended support faces instead.
- [ ] Gravity is enabled in Mechanical only if `fem.gravity_options` is not
  `None`.  Use `g_vec = fem.gravity_options.get('g_vec', [0, 0, -9.80665])`.

### Comparison locations

- [ ] Von Mises and B-field values are compared cell-by-cell at the
  **cell centroid** coordinates, not at nodes.
- [ ] ANSYS result averaging is set to **None** (element results, not
  nodal-smoothed) for the von Mises comparison.
- [ ] Nodal displacements are compared at the exact node coordinates from the
  VTU — use nearest-neighbour lookup (KD-tree) to verify 1-to-1 correspondence.

### Units

- [ ] All coordinates in metres (m).
- [ ] Currents in amperes (A).
- [ ] Stiffness in N/m³ (not N/mm³ or N/m²).
- [ ] Stress: coil-fem reports MPa; ANSYS default may be Pa — convert.
- [ ] B fields in tesla (T); Maxwell default is tesla.

### Student Edition size

- [ ] Mechanical structural mesh size stays within the Ansys Student structural
  physics limit: **128K nodes/elements** (about 300 coil quadpoints).  Check both `m.points.shape[0]`
  and the total VTU cell count before importing the mesh.
- [ ] Maxwell mesh element count in each solve stays ≤ **64 000 volume elements** (about 250 coil quadpoints).
  Use *Mesh Operations → Assign Mesh Operations → Surface Approximation* or
  *Initial Mesh Settings → Maximum Element Length* to control refinement.
- [ ] If a single full-assembly Maxwell solve exceeds 64 000 elements, split
  into two runs: (a) coil `i` alone → `B_self`, (b) all except coil `i` →
  `B_ext`.  Sum for `B_total`.
