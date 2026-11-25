# FreeCAD-SIMP: Native 3D Topology Optimization Engine

An in-memory 3D Solid Isotropic Material with Penalization (SIMP) topology optimization macro for FreeCAD. Directly couples with FreeCAD's native FEM Workbench data structures, avoiding external solver disk I/O, and reconstructs solid geometry via simplicial boundary extraction.

---

## Technical Highlights

- **In-Memory Pipeline:** Linear elasticity solves, adjoint filtering, and density updates remain resident in memory, eliminating external solver file parsing (`.inp`/`.frd`).
- **Vectorized C3D4 Formulation:** Analytical constant strain-displacement tensor assembly ($\mathbf{k}_0^e = V_e \mathbf{B}^T \mathbf{C} \mathbf{B}$) evaluated over linear tetrahedra using NumPy vectorization.
- **Adjoint Density Filter:** Implements volume-weighted physical density filtering ($\tilde{\boldsymbol{\rho}} = \mathbf{H}\boldsymbol{\rho}$) and adjoint sensitivity mapping ($\nabla_{\boldsymbol{\rho}} C = \mathbf{H}^T \nabla_{\tilde{\boldsymbol{\rho}}} C$) with filter radius clamped to $r_{\min} \ge 1.5 h_{\text{avg}}$ to prevent checkerboard instabilities.
- **Preconditioned CG Solver:** Solves linear systems using Preconditioned Conjugate Gradient with Jacobi preconditioning and displacement continuation ($\mathbf{x}_0 = \mathbf{u}^{(k-1)}$).
- **Design-Domain Volume Scaling:** Enforces mass reduction targets strictly on the optimizable design domain, preserving keep-in bodies without domain starvation or solver divergence.
- **Keep-In Domain Integration:** Identifies non-design features via dedicated FreeCAD Groups (`NonDesign`) or label keywords with global coordinate placement transforms and bounding-box accelerated containment tests.
- **Simplicial Boundary Extraction:** Extracts triangular facets from solid element boundaries, filters pinch singularities, heals micro-punctures, and applies volume-preserving Taubin smoothing.

---

## Mathematical Formulation

### 1. Linear Elasticity & Material Penalization
For linear tetrahedra (C3D4), material stiffness is parameterized by continuous element densities $\rho_e \in [\rho_{\min}, 1.0]$:

$$E(\tilde{\rho}_e) = E_{\min} + \tilde{\rho}_e^p (E_0 - E_{\min}), \quad \rho_{\min} = 10^{-3}, \; p \in [1, 3]$$

The global stiffness matrix is assembled over active degrees of freedom:

$$\mathbf{K}(\tilde{\boldsymbol{\rho}}) = \sum_{e=1}^{N_e} \frac{E(\tilde{\rho}_e)}{E_0} \mathbf{k}_0^e$$

### 2. Adjoint Density Filter
To enforce structural length scales on unstructured tetrahedral meshes with varying element volumes $V_e$, physical densities $\tilde{\boldsymbol{\rho}}$ are filtered from design variables $\boldsymbol{\rho}$:

$$\tilde{\rho}_i = \sum_{j \in \mathcal{N}_i} H_{ij} \rho_j, \quad H_{ij} = \frac{\max(0, r_{\min} - \|\mathbf{x}_c^i - \mathbf{x}_c^j\|) V_j}{\sum_{k \in \mathcal{N}_i} \max(0, r_{\min} - \|\mathbf{x}_c^i - \mathbf{x}_c^k\|) V_k}$$

The compliance sensitivity is mapped via the adjoint transpose:

$$\frac{\partial C}{\partial \boldsymbol{\rho}} = \mathbf{H}^T \frac{\partial C}{\partial \tilde{\boldsymbol{\rho}}}, \quad \text{where } \frac{\partial C}{\partial \tilde{\rho}_e} = -p \tilde{\rho}_e^{p-1} \left(1 - \frac{E_{\min}}{E_0}\right) \mathbf{u}_e^T \mathbf{k}_0^e \mathbf{u}_e$$

### 3. Optimality Criteria (OC) Update Scheme
A bisection search over the Lagrange multiplier $\lambda$ enforces the target physical volume fraction $V_f$ on the optimizable design domain:

$$B_e = \frac{-(\mathbf{H}^T \nabla_{\tilde{\boldsymbol{\rho}}} C)_e}{\lambda (\mathbf{H}^T \mathbf{V})_e}$$

$$\rho_e^{(k+1)} = \operatorname{clip}\Big( \operatorname{clip}\big( \rho_e^{(k)} B_e^\eta, \; \rho_e^{(k)} - m, \; \rho_e^{(k)} + m \big), \; \rho_{\min}, \; 1.0 \Big)$$

Where $\eta = 0.5$, and $m$ is the move limit with progressive decay ($m_{k+1} = \max(0.04, m_k \times 0.96)$ after iteration 15). Given non-design volume $V_{\text{frozen}} = \sum_{e \in \text{frozen}} V_e$ and design volume $V_{\text{design}} = V_{\text{total}} - V_{\text{frozen}}$, the target physical volume is:

$$V_{\text{target}} = V_{\text{frozen}} + V_f V_{\text{design}}$$

$$\sum_{e=1}^{N_e} (\mathbf{H}^T \mathbf{V})_e \rho_e \le V_{\text{target}}$$

### 4. Boundary Extraction & Mesh Fairing
Let $\mathcal{S} = \{e \mid \tilde{\rho}_e \ge \tau\} \cup \mathcal{S}_{\text{frozen}}$ denote the set of solid tetrahedra. Boundary facets belong to exactly one solid element:

$$\partial \mathcal{S} = \{ f \in \mathcal{F} \mid \operatorname{count}(f) = 1 \}$$

Singular pinch elements sharing fewer than two triangular faces with solid neighbors are pruned:

$$\mathcal{S}_{\text{pruned}} = \{ e \in \mathcal{S} \setminus \mathcal{S}_{\text{frozen}} \mid \operatorname{deg}_{\mathcal{G}}(e) \ge 2 \}$$

The extracted surface mesh is post-processed with Taubin volume-preserving smoothing:

$$\mathbf{x}^{(k+1)} = (\mathbf{I} + \mu \mathbf{L}) (\mathbf{I} + \lambda \mathbf{L}) \mathbf{x}^{(k)}, \quad 0 < \lambda < -\mu$$

---

## Repository Structure

```text
.
├── simp_optimizer.py            # Complete in-memory SIMP solver & mesh reconstructor
└── README.md                 # Technical documentation and workflow guide
```

---

## Installation & Deployment

Deploy the engine into your FreeCAD macro directory:

**Linux:**
```bash
cp simp_optimizer.py ~/.local/share/FreeCAD/Macro/SIMP_Optimizer.FCMacro
```

**macOS:**
```bash
cp simp_optimizer.py ~/Library/Application\ Support/FreeCAD/Macro/SIMP_Optimizer.FCMacro
```

**Windows (PowerShell):**
```powershell
Copy-Item simp_optimizer.py "$env:APPDATA\FreeCAD\Macro\SIMP_Optimizer.FCMacro"
```

Once copied, the engine appears under **Macro $\rightarrow$ Macros... $\rightarrow$ SIMP_Optimizer** and can be executed with **Execute** or assigned to a custom toolbar button.

---

## Workflow Guide

### Step 1: CAD Preparation (Design vs. Non-Design Domains)
1. Model the packaging volume (the optimizable design space).
2. Model keep-in solid bodies (mounting bushings, bearing seats) representing regions that must remain 100% solid.
   - **Method 1 (Recommended):** Create a FreeCAD Group named `NonDesign` and move keep-in solids into it.
   - **Method 2:** Label keep-in objects with an identifiable keyword (e.g. `Body_NonDesign`, `KeepIn`, `Preserve`, `Frozen`).
3. Switch to the **Part Workbench**.
4. Select the design space body, hold `Ctrl`, select the keep-in body, and execute:  
   **Part $\rightarrow$ Split $\rightarrow$ Slice to compound**  
   *(Partitions the volumes along conforming shared interfaces).*

### Step 2: FEM Setup
1. Switch to the **FEM Workbench** and click **New Analysis Container**.
2. Place the sliced compound into the `FemAnalysis` container.
3. Assign a solid material via **FEM Material for Solid** (Young's modulus and Poisson's ratio are ingested automatically).
4. Assign boundary conditions:
   - **Supports:** Add `Fem::ConstraintFixed` or `Fem::ConstraintDisplacement`. Individual fixed Cartesian axes ($X, Y, Z$) are respected, permitting symmetry planes and roller supports.
   - **Loads:** Add `Fem::ConstraintPressure` or `Fem::ConstraintForce`. If `DirectionVector` is unset, force automatically aligns along the selected face normal.
5. Mesh the compound with **Gmsh**:
   - Element Dimension: `3D`.
   - **Element Order:** `1` (Linear tetrahedra TET4 required).
   - Max Element Size: Set based on target feature resolution ($h \approx \frac{\text{Wall Thickness}}{3 \dots 5}$).

### Step 3: Optimization Execution
1. Open `SIMP_Optimizer.FCMacro` in FreeCAD (**Macro $\rightarrow$ Macros... $\rightarrow$ Edit**).
2. Adjust configuration parameters in `CONFIG` if needed (e.g., target volume fraction).
3. Click **Execute** (or press **Ctrl + F6**).
4. Monitor numerical convergence in the Report View and track live density evolution in the 3D viewport.

### Step 4: Export for Additive Manufacturing
1. Upon convergence, the optimized solid mesh is generated as `SIMP_Final_Printable_Mesh`.
2. Select the object in the tree view.
3. Navigate to **File $\rightarrow$ Export** and select **3MF file (*.3mf)** or **STL Mesh (*.stl)**.

---

## Configuration Reference (`CONFIG`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_volume_fraction` | `float` | `0.35` | Target retained volume fraction of the optimizable design domain ($0.35 = 35\%$). |
| `min_member_thickness_mm` | `float` | `12.0` | Minimum strut diameter. Clamped internally to $r_{\min} \ge 1.5 h_{\text{avg}}$. |
| `density_cutoff_tau` | `float` | `0.52` | Iso-density boundary extraction threshold ($0.45\text{--}0.55$). |
| `nondesign_group` | `str` | `"NonDesign"` | Preferred FreeCAD Group name for keep-in solids. |
| `nondesign_keywords` | `list` | `[...]` | Fallback keywords for keep-in object labels (`"nondesign"`, `"keep"`, `"preserve"`, `"frozen"`, `"non_design"`). |
| `support_collar_radius_mm`| `float` | `3.0` | Safety collar radius around fixed nodes ($0.0$ disables; $>0$ maintains solid anchor seeds). |
| `load_collar_radius_mm` | `float` | `1.5` | Safety collar radius under applied load nodes ($0.0$ disables). |
| `enable_live_viewport` | `bool` | `True` | Set `False` to disable interactive viewport updates for solve speed. |
| `live_update_interval` | `int` | `1` | Viewport refresh frequency (iterations between redraws). |
| `max_micro_hole_edges` | `int` | `12` | Max boundary loop edges to heal ($0$ disables). |
| `min_component_facets` | `int` | `20` | Minimum facet count to retain disconnected solid shells. |
| `taubin_smooth_passes` | `int` | `18` | Volume-preserving Taubin smoothing passes. |
| `anti_pinch_passes` | `int` | `2` | Face-neighbor filter iterations to prune 1-element pinch singularities. |
| `max_iterations` | `int` | `60` | Solver iteration ceiling. |
| `convergence_tolerance` | `float` | `0.005` | Convergence threshold on maximum element density change ($\Delta \rho < \text{tol}$). |
| `move_limit_initial` | `float` | `0.20` | Initial design variable move limit per iteration. |
| `pcg_tolerance` | `float` | `1e-3` | Relative residual stopping tolerance for warm-started PCG solver. |

---

## Mathematical Foundations & References

1. **SIMP Formulation & Optimality Criteria (OC):**  
   Bendsøe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications*. Springer Berlin Heidelberg.
2. **Volume-Weighted Adjoint Density Filtering:**  
   Bourdin, B. (2001). *Filters in topology optimization*. International Journal for Numerical Methods in Engineering, 50(9), 2143–2158.  
   Bruns, T. E., & Tortorelli, D. A. (2001). *Topology optimization of non-linear elastic structures and compliant mechanisms*. Computer Methods in Applied Mechanics and Engineering, 190(26–27), 3443–3459.
3. **Volume-Preserving Mesh Fairing (Taubin Filter):**  
   Taubin, G. (1995). *Curve and surface smoothing without shrinkage*. In Proceedings of IEEE International Conference on Computer Vision (ICCV '95), 852–857.
