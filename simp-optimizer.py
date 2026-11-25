"""
FreeCAD 3D SIMP Topology Optimization Engine
In-memory linear elasticity solver and simplicial boundary extraction for FreeCAD FEM Workbench meshes.
"""

import inspect
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import FreeCAD as App
import FreeCADGui as Gui
import Mesh

CONFIG = {
    # Retained solid volume fraction of the optimizable design domain (0.35 = 35%)
    "target_volume_fraction": 0.35,

    # Minimum strut diameter (mm). Clamped internally to r_min >= 1.5 * h_avg
    "min_member_thickness_mm": 12.0,

    # Iso-density boundary extraction cutoff (tau in [0.45, 0.55])
    "density_cutoff_tau": 0.52,

    # Non-design domain identification
    "nondesign_group": "NonDesign",
    "nondesign_keywords": ["nondesign", "keep", "preserve", "frozen", "non_design"],

    # Safety collars around boundary nodes in mm (0.0 disables; >0 maintains solid anchor seeds)
    "support_collar_radius_mm": 3.0,
    "load_collar_radius_mm": 1.5,

    # --- Live Viewport & Performance Controls ---
    "enable_live_viewport": True,        # Set False to disable live rendering for maximum solve speed
    "live_update_interval": 1,           # Viewport refresh rate (iterations between redraws)

    # --- Mesh Reconstruction & Repair ---
    "max_micro_hole_edges": 12,          # Max boundary loop edges to heal (0 disables)
    "min_component_facets": 20,          # Cull detached floating debris smaller than N facets
    "taubin_smooth_passes": 18,          # Volume-preserving fairing iterations
    "anti_pinch_passes": 2,              # Topological filter passes to prune singular pinch elements

    # --- Optimization Stopping Criteria ---
    "max_iterations": 60,
    "convergence_tolerance": 0.005,
    "move_limit_initial": 0.20,
    "pcg_tolerance": 1e-3,
}


def build_element_connectivity(elems, num_elems):
    """Finds adjacent tetrahedron pairs (e1, e2) sharing a triangular face."""
    all_tet_faces = np.vstack([
        elems[:, [0, 2, 1]],
        elems[:, [0, 1, 3]],
        elems[:, [1, 2, 3]],
        elems[:, [0, 3, 2]]
    ])
    elem_of_face = np.tile(np.arange(num_elems, dtype=np.int32), 4)

    sorted_faces = np.sort(all_tet_faces, axis=1)
    sort_order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
    sorted_faces = sorted_faces[sort_order]
    elem_of_face = elem_of_face[sort_order]

    matches = np.all(sorted_faces[:-1] == sorted_faces[1:], axis=1)
    e1 = elem_of_face[:-1][matches]
    e2 = elem_of_face[1:][matches]
    return e1, e2


def filter_anti_pinch(solid_mask, frozen_mask, e1, e2, num_elems, passes):
    """Prunes isolated solid elements sharing fewer than 2 faces with solid neighbors."""
    pruned_mask = solid_mask.copy()
    for _ in range(passes):
        shared_solid = pruned_mask[e1] & pruned_mask[e2]
        num_neighbors = np.zeros(num_elems, dtype=np.int32)
        np.add.at(num_neighbors, e1[shared_solid], 1)
        np.add.at(num_neighbors, e2[shared_solid], 1)

        to_prune = pruned_mask & (~frozen_mask) & (num_neighbors < 2)
        if not np.any(to_prune):
            break
        pruned_mask[to_prune] = False
    return pruned_mask


def extract_boundary_faces(coords, elems, solid_mask):
    """Extracts outward-oriented triangular boundary facets from solid tetrahedra."""
    solid_elems = elems[solid_mask]
    if len(solid_elems) == 0:
        return np.empty((0, 3, 3), dtype=np.float64)

    faces = np.vstack([
        solid_elems[:, [0, 2, 1]],
        solid_elems[:, [0, 1, 3]],
        solid_elems[:, [1, 2, 3]],
        solid_elems[:, [0, 3, 2]]
    ])
    sorted_faces = np.sort(faces, axis=1)
    _, u_idx, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    boundary_nodes = faces[u_idx[counts == 1]]
    return coords[boundary_nodes]


def build_live_mesh(coords, elems, rho_phys, frozen_mask, e1, e2, num_elems, tau, passes):
    """Extracts boundary facets for interactive viewport re-draws."""
    solid_mask = (rho_phys >= tau) | frozen_mask
    solid_mask = filter_anti_pinch(solid_mask, frozen_mask, e1, e2, num_elems, passes)
    triangles = extract_boundary_faces(coords, elems, solid_mask)
    if len(triangles) == 0:
        return Mesh.Mesh()
    m = Mesh.Mesh(triangles.tolist())
    m.harmonizeNormals()
    return m


def finalize_watertight_mesh(coords, elems, rho_phys, frozen_mask, e1, e2, num_elems, tau, smooth_passes, pinch_passes, max_hole_edges, min_components):
    """Watertight surface post-processing via simplicial extraction, micro-hole healing, and Taubin smoothing."""
    solid_mask = (rho_phys >= tau) | frozen_mask
    solid_mask = filter_anti_pinch(solid_mask, frozen_mask, e1, e2, num_elems, pinch_passes)
    triangles = extract_boundary_faces(coords, elems, solid_mask)
    if len(triangles) == 0:
        return Mesh.Mesh()

    m = Mesh.Mesh(triangles.tolist())

    if min_components > 0:
        m.removeComponents(min_components)

    if max_hole_edges > 0:
        m.fillupHoles(max_hole_edges)

    if smooth_passes > 0:
        m.smooth("Taubin", smooth_passes)

    m.harmonizeNormals()
    return m


def run_simp():
    t_start = time.perf_counter()
    print("--- FreeCAD SIMP Topology Optimization Engine ---")

    doc = App.ActiveDocument
    if not doc:
        App.Console.PrintError("SIMP Error: No active document found.\n")
        return

    analyses = [o for o in doc.Objects if o.isDerivedFrom("Fem::FemAnalysis")]
    if not analyses:
        App.Console.PrintError("SIMP Error: No FemAnalysis container found in active document.\n")
        return
    analysis = analyses[0]

    mesh_objs = [o for o in analysis.Group if o.isDerivedFrom("Fem::FemMeshObject")]
    if not mesh_objs:
        App.Console.PrintError("SIMP Error: No FemMesh found in Analysis container.\n")
        return
    fem_mesh = mesh_objs[0].FemMesh

    # 1. Ingest Nodes and Elements
    t0 = time.perf_counter()
    raw_nodes = fem_mesh.Nodes
    node_ids = sorted(raw_nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    coords = np.array([[raw_nodes[nid].x, raw_nodes[nid].y, raw_nodes[nid].z] for nid in node_ids], dtype=np.float64)

    elem_ids = sorted(fem_mesh.Volumes)
    if not elem_ids:
        App.Console.PrintError("SIMP Error: FemMesh contains no 3D solid volume elements.\n")
        return

    raw_elem_nodes = [fem_mesh.getElementNodes(eid) for eid in elem_ids]
    if any(len(n) > 4 for n in raw_elem_nodes[:20]):
        App.Console.PrintError("SIMP Error: Quadratic elements (TET10) detected. Set Gmsh element order to 1 (linear).\n")
        return

    elems = np.array([[nid_to_idx[n] for n in en[:4]] for en in raw_elem_nodes], dtype=np.int32)
    num_nodes, num_elems = len(node_ids), len(elem_ids)
    num_dofs = num_nodes * 3
    centroids = coords[elems].mean(axis=1)

    # 2. Positive Tet Orientations & Volume
    M = np.ones((num_elems, 4, 4), dtype=np.float64)
    M[:, :, 1:] = coords[elems]
    det_M = np.linalg.det(M)
    neg_tets = det_M < 0
    if np.any(neg_tets):
        elems[neg_tets, 1], elems[neg_tets, 2] = elems[neg_tets, 2].copy(), elems[neg_tets, 1].copy()
        M[neg_tets, :, 1:] = coords[elems[neg_tets]]
        det_M = np.linalg.det(M)

    V_e = det_M / 6.0
    if np.any(V_e <= 0):
        App.Console.PrintError("SIMP Error: Mesh contains degenerate or zero-volume elements.\n")
        return

    V_total = float(np.sum(V_e))
    h_avg = float(np.mean(V_e) ** (1.0 / 3.0))
    e1_adj, e2_adj = build_element_connectivity(elems, num_elems)
    print(f"[Setup] Mesh: {num_nodes:,} nodes | {num_elems:,} elements | h_avg = {h_avg:.2f} mm ({time.perf_counter()-t0:.2f}s)")

    # 3. Dynamic Material Ingestion
    def is_solid_material(o):
        if getattr(o, "Category", "Solid") == "Fluid":
            return False
        proxy_type = getattr(getattr(o, "Proxy", None), "Type", "")
        if proxy_type in ("Fem::MaterialSolid", "Fem::MaterialCommon", "MaterialSolid"):
            return True
        if o.isDerivedFrom("App::MaterialObjectPython") or o.isDerivedFrom("App::MaterialObject"):
            return True
        return hasattr(o, "Material")

    mat_objs = [o for o in analysis.Group if is_solid_material(o)]
    if not mat_objs:
        mat_objs = [o for o in doc.Objects if o not in analysis.Group and is_solid_material(o)]

    E0, nu = 3000.0, 0.36
    if mat_objs:
        mat_obj = mat_objs[0]
        props = getattr(mat_obj, "Material", {})
        ym, pr = None, None
        if isinstance(props, dict):
            ym = props.get("YoungsModulus", "")
            pr = props.get("PoissonRatio", "")
        else:
            ym = getattr(props, "YoungsModulus", getattr(mat_obj, "YoungsModulus", ""))
            pr = getattr(props, "PoissonRatio", getattr(mat_obj, "PoissonRatio", ""))

        try:
            if ym:
                if hasattr(ym, "getValueAs"):
                    E0 = float(ym.getValueAs("MPa"))
                elif isinstance(ym, str):
                    E0 = float(App.Units.Quantity(ym).getValueAs("MPa"))
                else:
                    E0 = float(ym)
            if pr:
                if hasattr(pr, "Value"):
                    nu = float(pr.Value)
                elif isinstance(pr, str):
                    nu = float(App.Units.Quantity(pr).Value)
                else:
                    nu = float(pr)
            print(f"[Setup] Material: '{mat_obj.Label}' (E = {E0:.1f} MPa, nu = {nu:.3f})")
        except Exception as e:
            print(f"[Setup] Material parse warning: {e}. Defaulting to E = {E0:.1f} MPa, nu = {nu:.3f}")
    else:
        print(f"[Setup] Default Material Assigned: E = {E0:.1f} MPa, nu = {nu:.3f}")

    if nu >= 0.499:
        nu = 0.49
    Emin = 1e-3 * E0

    lam = (E0 * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E0 / (2.0 * (1.0 + nu))
    C_mat = np.zeros((6, 6), dtype=np.float64)
    C_mat[0, 0] = C_mat[1, 1] = C_mat[2, 2] = lam + 2.0 * mu
    C_mat[0, 1] = C_mat[0, 2] = C_mat[1, 0] = C_mat[1, 2] = C_mat[2, 0] = C_mat[2, 1] = lam
    C_mat[3, 3] = C_mat[4, 4] = C_mat[5, 5] = mu

    # 4. Element Stiffness Assembly (Linear Tetrahedron C3D4)
    invM = np.linalg.inv(M)
    b, c, d = invM[:, 1, :], invM[:, 2, :], invM[:, 3, :]
    B = np.zeros((num_elems, 6, 12), dtype=np.float64)
    for i in range(4):
        col = 3 * i
        B[:, 0, col + 0] = b[:, i]
        B[:, 1, col + 1] = c[:, i]
        B[:, 2, col + 2] = d[:, i]
        B[:, 3, col + 0] = c[:, i]
        B[:, 3, col + 1] = b[:, i]
        B[:, 4, col + 1] = d[:, i]
        B[:, 4, col + 2] = c[:, i]
        B[:, 5, col + 0] = d[:, i]
        B[:, 5, col + 2] = b[:, i]

    CB = np.matmul(C_mat[None, :, :], B)
    k0 = V_e[:, None, None] * np.matmul(np.swapaxes(B, 1, 2), CB)

    elem_dofs = np.zeros((num_elems, 12), dtype=np.int32)
    for i in range(4):
        elem_dofs[:, 3*i + 0] = 3 * elems[:, i] + 0
        elem_dofs[:, 3*i + 1] = 3 * elems[:, i] + 1
        elem_dofs[:, 3*i + 2] = 3 * elems[:, i] + 2

    rows = np.repeat(elem_dofs[:, :, None], 12, axis=2).reshape(-1)
    cols = np.repeat(elem_dofs[:, None, :], 12, axis=1).reshape(-1)

    # 5. Boundary Conditions (Supports and Loads)
    fixed_dofs_set = set()
    for o in analysis.Group:
        if o.isDerivedFrom("Fem::ConstraintFixed") or o.isDerivedFrom("Fem::ConstraintDisplacement"):
            is_disp = o.isDerivedFrom("Fem::ConstraintDisplacement")
            x_fixed = (not getattr(o, "xFree", False)) and getattr(o, "xFix", True) if is_disp else True
            y_fixed = (not getattr(o, "yFree", False)) and getattr(o, "yFix", True) if is_disp else True
            z_fixed = (not getattr(o, "zFree", False)) and getattr(o, "zFix", True) if is_disp else True

            c_node_ids = set()
            for owner, subs in o.References:
                subs_list = [subs] if isinstance(subs, str) else subs
                for s in subs_list:
                    elem = owner.Shape.getElement(s)
                    if elem.ShapeType == "Face":
                        c_node_ids.update(fem_mesh.getNodesByFace(elem))
                    elif elem.ShapeType == "Edge":
                        c_node_ids.update(fem_mesh.getNodesByEdge(elem))
                    elif elem.ShapeType == "Vertex":
                        v_coords = np.array([elem.Point.x, elem.Point.y, elem.Point.z])
                        dist = np.linalg.norm(coords - v_coords, axis=1)
                        if len(dist) > 0:
                            c_node_ids.add(node_ids[np.argmin(dist)])

            for nid in c_node_ids:
                if nid in nid_to_idx:
                    idx = nid_to_idx[nid]
                    if x_fixed:
                        fixed_dofs_set.add(3 * idx + 0)
                    if y_fixed:
                        fixed_dofs_set.add(3 * idx + 1)
                    if z_fixed:
                        fixed_dofs_set.add(3 * idx + 2)

    fixed_dofs = np.array(sorted(fixed_dofs_set), dtype=np.int32)
    if fixed_dofs.size == 0:
        App.Console.PrintError("SIMP Error: No fixed support nodes found. Structure must be constrained.\n")
        return

    free_dofs = np.setdiff1d(np.arange(num_dofs, dtype=np.int32), fixed_dofs)
    fixed_indices = np.unique(fixed_dofs // 3)

    f_global = np.zeros(num_dofs, dtype=np.float64)
    loaded_node_ids = set()

    for o in analysis.Group:
        if o.isDerivedFrom("Fem::ConstraintPressure"):
            if isinstance(o.Pressure, str):
                p_val = float(App.Units.Quantity(o.Pressure).getValueAs("MPa"))
            elif hasattr(o.Pressure, "getValueAs"):
                p_val = float(o.Pressure.getValueAs("MPa"))
            elif hasattr(o.Pressure, "Value"):
                p_val = float(o.Pressure.Value)
            else:
                p_val = float(o.Pressure)
            sign = 1.0 if getattr(o, "Reversed", False) else -1.0

            for owner, subs in o.References:
                subs_list = [subs] if isinstance(subs, str) else subs
                for s in subs_list:
                    face = owner.Shape.getElement(s)
                    p_nodes = [nid_to_idx[nid] for nid in fem_mesh.getNodesByFace(face) if nid in nid_to_idx]
                    if not p_nodes:
                        continue
                    loaded_node_ids.update(p_nodes)

                    u0, u1, v0, v1 = face.ParameterRange
                    norm = face.normalAt((u0 + u1) * 0.5, (v0 + v1) * 0.5)
                    norm_vec = np.array([norm.x, norm.y, norm.z], dtype=np.float64)
                    norm_mag = np.linalg.norm(norm_vec)
                    if norm_mag > 1e-12:
                        norm_vec /= norm_mag

                    area = float(face.Area)
                    f_total = sign * p_val * area * norm_vec
                    f_per_node = f_total / len(p_nodes)
                    for nid in p_nodes:
                        f_global[3 * nid : 3 * nid + 3] += f_per_node

        elif o.isDerivedFrom("Fem::ConstraintForce"):
            if isinstance(o.Force, str):
                f_val = float(App.Units.Quantity(o.Force).getValueAs("N"))
            elif hasattr(o.Force, "getValueAs"):
                f_val = float(o.Force.getValueAs("N"))
            elif hasattr(o.Force, "Value"):
                f_val = float(o.Force.Value) / 1000.0
            else:
                f_val = float(o.Force)
            sign = -1.0 if getattr(o, "Reversed", False) else 1.0

            d_vec = None
            if hasattr(o, "DirectionVector"):
                dv = o.DirectionVector
                cand = np.array([dv.x, dv.y, dv.z], dtype=np.float64)
                mag = np.linalg.norm(cand)
                if mag > 1e-12:
                    d_vec = cand / mag

            constraint_nodes = set()
            fallback_face_normal = None

            for owner, subs in o.References:
                subs_list = [subs] if isinstance(subs, str) else subs
                for s in subs_list:
                    elem = owner.Shape.getElement(s)
                    if elem.ShapeType == "Face":
                        constraint_nodes.update(nid_to_idx[nid] for nid in fem_mesh.getNodesByFace(elem) if nid in nid_to_idx)
                        if fallback_face_normal is None:
                            u0, u1, v0, v1 = elem.ParameterRange
                            n = elem.normalAt((u0 + u1) * 0.5, (v0 + v1) * 0.5)
                            n_vec = np.array([n.x, n.y, n.z], dtype=np.float64)
                            n_mag = np.linalg.norm(n_vec)
                            if n_mag > 1e-12:
                                fallback_face_normal = n_vec / n_mag
                    elif elem.ShapeType == "Edge":
                        constraint_nodes.update(nid_to_idx[nid] for nid in fem_mesh.getNodesByEdge(elem) if nid in nid_to_idx)
                    elif elem.ShapeType == "Vertex":
                        v_coords = np.array([elem.Point.x, elem.Point.y, elem.Point.z])
                        constraint_nodes.add(np.argmin(np.linalg.norm(coords - v_coords, axis=1)))

            if not constraint_nodes:
                continue

            if d_vec is None:
                if fallback_face_normal is not None:
                    d_vec = fallback_face_normal
                else:
                    d_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                    print(f"[Setup] Note: ConstraintForce '{o.Label}' has no direction vector or face normal. Defaulting to -Z.")

            loaded_node_ids.update(constraint_nodes)
            f_per_node = (sign * f_val * d_vec) / len(constraint_nodes)
            for nid in constraint_nodes:
                f_global[3 * nid : 3 * nid + 3] += f_per_node

    f_free = f_global[free_dofs]
    if np.linalg.norm(f_free) == 0:
        App.Console.PrintError("SIMP Error: Total external force vector is zero. Apply at least one active load or pressure.\n")
        return

    # 6. Non-Design Keep-In Volumes
    t0_tag = time.perf_counter()
    frozen_elem_set = set()
    nd_candidates = []

    # Method 1: Dedicated FreeCAD Group
    grp_name = CONFIG.get("nondesign_group", "NonDesign")
    grp = doc.getObject(grp_name)
    if grp and hasattr(grp, "Group"):
        nd_candidates.extend([o for o in grp.Group if hasattr(o, "Shape") and o.Shape])

    # Method 2: Label keyword matching
    kw = [k.lower() for k in CONFIG.get("nondesign_keywords", [])]
    if kw:
        for o in doc.Objects:
            if hasattr(o, "Shape") and o.Shape and o not in nd_candidates:
                if any(k in o.Label.lower() for k in kw):
                    nd_candidates.append(o)

    for nd_obj in nd_candidates:
        shape = nd_obj.Shape.copy()
        if hasattr(nd_obj, "getGlobalPlacement"):
            shape.Placement = nd_obj.getGlobalPlacement()
        solids = shape.Solids if shape.Solids else [shape]
        for solid in solids:
            bb = solid.BoundBox
            in_bb = (
                (centroids[:, 0] >= bb.XMin - 0.5) & (centroids[:, 0] <= bb.XMax + 0.5) &
                (centroids[:, 1] >= bb.YMin - 0.5) & (centroids[:, 1] <= bb.YMax + 0.5) &
                (centroids[:, 2] >= bb.ZMin - 0.5) & (centroids[:, 2] <= bb.ZMax + 0.5)
            )
            for idx in np.where(in_bb)[0]:
                if solid.isInside(App.Vector(centroids[idx, 0], centroids[idx, 1], centroids[idx, 2]), 0.1, True):
                    frozen_elem_set.add(idx)

    tree = cKDTree(centroids)
    if fixed_indices.size > 0 and CONFIG["support_collar_radius_mm"] > 0:
        for sub in tree.query_ball_point(coords[fixed_indices], r=CONFIG["support_collar_radius_mm"]):
            frozen_elem_set.update(sub)
    if len(loaded_node_ids) > 0 and CONFIG["load_collar_radius_mm"] > 0:
        for sub in tree.query_ball_point(coords[list(loaded_node_ids)], r=CONFIG["load_collar_radius_mm"]):
            frozen_elem_set.update(sub)

    frozen_mask = np.zeros(num_elems, dtype=bool)
    frozen_mask[list(frozen_elem_set)] = True
    V_frozen = float(np.sum(V_e[frozen_mask]))
    V_design = V_total - V_frozen
    print(f"[Setup] Frozen Non-Design: {np.sum(frozen_mask):,} / {num_elems:,} elements ({V_frozen/V_total*100:.1f}% volume) in {time.perf_counter()-t0_tag:.2f}s")

    # 7. Mesh-Clamped Adjoint Filter Matrix
    t0 = time.perf_counter()
    r_min = max(1.5 * h_avg, CONFIG["min_member_thickness_mm"] / 2.0)
    neighbor_lists = tree.query_ball_point(centroids, r=r_min)

    filter_rows, filter_cols, filter_weights = [], [], []
    for i, nbrs in enumerate(neighbor_lists):
        nbrs = np.array(nbrs, dtype=np.int32)
        dist = np.linalg.norm(centroids[i] - centroids[nbrs], axis=1)
        w = np.maximum(0.0, r_min - dist) * V_e[nbrs]
        sum_w = np.sum(w)
        if sum_w > 0:
            w /= sum_w
        filter_rows.extend([i] * len(nbrs))
        filter_cols.extend(nbrs)
        filter_weights.extend(w)

    H_filter = sp.csr_matrix((filter_weights, (filter_rows, filter_cols)), shape=(num_elems, num_elems))
    dV_drho = H_filter.T.dot(V_e)
    print(f"[Setup] Filter Matrix Built: r_min = {r_min:.2f} mm (clamped >= 1.5 * h_avg) in {time.perf_counter()-t0:.2f}s")

    # 8. Viewport Object Initialization
    live_obj = doc.getObject("SIMP_Final_Printable_Mesh") or doc.addObject("Mesh::Feature", "SIMP_Final_Printable_Mesh")
    live_obj.Label = "SIMP_Final_Printable_Mesh"
    if getattr(App, "GuiUp", False):
        live_obj.ViewObject.ShapeColor = (0.2, 0.7, 0.9)
        live_obj.ViewObject.DisplayMode = "Shaded"

    # 9. Optimization Loop
    Vf_target = CONFIG["target_volume_fraction"]
    V_target_total = V_frozen + Vf_target * V_design
    max_iters = CONFIG["max_iterations"]
    tol = CONFIG["convergence_tolerance"]
    move = CONFIG["move_limit_initial"]
    live_interval = max(1, int(CONFIG.get("live_update_interval", 1)))
    enable_live = bool(CONFIG.get("enable_live_viewport", True)) and bool(getattr(App, "GuiUp", False))

    rho = np.full(num_elems, Vf_target, dtype=np.float64)
    rho[frozen_mask] = 1.0
    rho_phys = H_filter.dot(rho)

    u_free = np.zeros(len(free_dofs), dtype=np.float64)
    u_global = np.zeros(num_dofs, dtype=np.float64)

    cg_kwargs = {"rtol": CONFIG["pcg_tolerance"]} if "rtol" in inspect.signature(spla.cg).parameters else {"tol": CONFIG["pcg_tolerance"]}

    if enable_live:
        live_obj.Mesh = build_live_mesh(coords, elems, rho_phys, frozen_mask, e1_adj, e2_adj, num_elems, CONFIG["density_cutoff_tau"], CONFIG["anti_pinch_passes"])
        doc.recompute()
        if App.GuiUp:
            Gui.updateGui()

    print("\n--- Starting Optimization Loop ---")
    print("Iter | Penal (p) | Compliance (N*mm) | Vol Frac | Delta Rho | PCG Iters | Move | Time")

    for it in range(1, max_iters + 1):
        t_iter0 = time.perf_counter()
        p = 1.0 if it <= 5 else (2.0 if it <= 10 else 3.0)

        if it > 15:
            move = max(0.04, move * 0.96)

        E_e = Emin + (rho_phys ** p) * (E0 - Emin)
        scale = (E_e / E0)[:, None, None]
        vals = (k0 * scale).reshape(-1)

        K_global = sp.coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()
        K_free = K_global[free_dofs, :][:, free_dofs]

        diag_K = K_free.diagonal()
        diag_K[diag_K == 0] = 1.0
        M_prec = sp.linalg.LinearOperator(K_free.shape, matvec=lambda x: (1.0 / diag_K) * x)

        pcg_count = 0
        def cg_monitor(_):
            nonlocal pcg_count
            pcg_count += 1

        u_free, info = spla.cg(K_free, f_free, x0=u_free, M=M_prec, maxiter=2500, callback=cg_monitor, **cg_kwargs)
        if info > 0:
            App.Console.PrintWarning(f"SIMP Warning: PCG did not reach tolerance at iteration {it} (code: {info}).\n")
        u_global[free_dofs] = u_free

        u_e = u_global[elem_dofs]
        k0_u = np.einsum('eij,ej->ei', k0, u_e)
        ce_base = np.einsum('ei,ei->e', u_e, k0_u)
        compliance = float(np.sum((Emin + (rho_phys ** p) * (E0 - Emin)) / E0 * ce_base))

        dC_drho_phys = -p * (rho_phys ** (p - 1)) * (1.0 - Emin / E0) * ce_base
        dC_drho = H_filter.T.dot(dC_drho_phys)

        l1, l2 = 1e-12, 1e12
        rho_old = rho.copy()

        # Bisection search over Lagrange multiplier lambda
        while (l2 - l1) / (l1 + l2) > 1e-4:
            l_mid = 0.5 * (l1 + l2)
            Be = -dC_drho / (l_mid * dV_drho)

            rho_cand = rho * np.sqrt(np.maximum(1e-12, Be))
            rho_new = np.clip(np.clip(rho_cand, rho - move, rho + move), 1e-3, 1.0)
            rho_new[frozen_mask] = 1.0

            if np.sum(rho_new * dV_drho) > V_target_total:
                l1 = l_mid
            else:
                l2 = l_mid

        rho = rho_new
        rho_phys = H_filter.dot(rho)

        delta_rho = float(np.max(np.abs(rho - rho_old)))
        curr_vf = float((np.sum(rho_phys * V_e) - V_frozen) / V_design) if V_design > 0 else float(np.sum(rho_phys * V_e) / V_total)

        # Interactive live viewport update (bypassed if enable_live_viewport is False or headless)
        if enable_live and (it % live_interval == 0 or it == max_iters):
            live_obj.Mesh = build_live_mesh(coords, elems, rho_phys, frozen_mask, e1_adj, e2_adj, num_elems, CONFIG["density_cutoff_tau"], CONFIG["anti_pinch_passes"])
            doc.recompute()
            if App.GuiUp:
                Gui.updateGui()

        t_iter = time.perf_counter() - t_iter0
        print(f"{it:4d} | {p:9.1f} | {compliance:17.4e} | {curr_vf:8.3f} | {delta_rho:9.4f} | {pcg_count:9d} | {move:4.2f} | {t_iter:.2f}s")

        if it > 20 and delta_rho < tol:
            print(f"--> Convergence achieved at iteration {it} (Delta Rho < {tol})")
            break

    # 10. Final Solid Solidification & Mesh Repair
    print("\n[Post-Process] Finalizing watertight solid...")
    min_comps = CONFIG.get("min_component_facets", CONFIG.get("remove_components_min_facets", 20))
    final_mesh = finalize_watertight_mesh(
        coords, elems, rho_phys, frozen_mask, e1_adj, e2_adj, num_elems,
        CONFIG["density_cutoff_tau"], CONFIG["taubin_smooth_passes"], CONFIG["anti_pinch_passes"],
        CONFIG["max_micro_hole_edges"], min_comps
    )
    live_obj.Mesh = final_mesh
    doc.recompute()
    if getattr(App, "GuiUp", False):
        Gui.updateGui()

    comps = final_mesh.getSeparateComponents()
    print(f"[Verification] Facets: {final_mesh.CountFacets:,} | Watertight Solid: {final_mesh.isSolid()} | Disconnected Shells: {len(comps)}")
    print(f"SIMP optimization finished in {time.perf_counter()-t_start:.2f}s.")
    return final_mesh


if __name__ == "__main__":
    run_simp()

