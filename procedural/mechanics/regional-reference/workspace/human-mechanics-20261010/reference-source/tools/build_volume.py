"""Full code-authored skin FEM reduction, including scalp, face and pinnae.
Topology-preserving QEM reduction retains actual authored vertices; no imported skin or sampled face dataset.
"""

from pathlib import Path
import json, hashlib, time, subprocess
import numpy as np, trimesh
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gltf(v):
    return v[:, [0, 2, 1]] * [1, 1, -1]


def build(source_path, output_path, shape_path, target_faces=4500):
    OUT = Path(output_path)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = Path(source_path)
    s = np.load(source)
    shape_file = Path(shape_path)
    shape = json.loads(shape_file.read_text(encoding="utf-8"))
    points = []
    triangles = []
    depths = []
    dermals = []
    groups = []
    report = []
    original = s["0_v"].astype(float)
    faces = s["0_f"]
    component = s["0_component_ids"]
    source_delta = np.fromfile(ROOT / shape["parts"][0]["url"], dtype="<f4").reshape(
        -1, 3
    )
    shaped = gltf(original) + source_delta
    for group in np.unique(component):
        f = faces[component == group]
        used = np.unique(f)
        v = original[used]
        f = np.searchsorted(used, f)
        boundary_edges, boundary_counts = np.unique(
            np.sort(np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1),
            axis=0,
            return_counts=True,
        )
        protected = np.zeros(len(v), bool)
        protected[np.unique(boundary_edges[boundary_counts == 1])] = True
        pinna = (
            (abs(v[:, 0]) > 0.067)
            & (v[:, 1] > 0)
            & (v[:, 2] > -0.030)
            & (v[:, 2] < 0.045)
        )
        protected |= pinna
        # A coarse coordinate weld must not close a real lip/eye/nasal gap or
        # merge opposite auricular surfaces. Only smooth interior aliases use
        # the coarse grid; aperture and thin-feature vertices keep their positions.
        rounded = np.round(v / 0.00005) * 0.00005
        rounded[protected] = v[protected]
        keys = np.c_[rounded, protected.astype(float)]
        keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        unique = keys[:, :3]
        f = inverse[f]
        valid = (f[:, 0] != f[:, 1]) & (f[:, 0] != f[:, 2]) & (f[:, 1] != f[:, 2])
        f = f[valid]
        print(
            json.dumps(
                {
                    "stage": "topology_preserving_reduction",
                    "component": int(group),
                    "vertices": len(unique),
                    "faces": len(f),
                }
            ),
            flush=True,
        )
        # Only edge collapses admitted by meshoptimizer; topology stays; boundary edge reduction is bounded by the error metric.
        tree = cKDTree(v)
        _, u_index = tree.query(unique)
        normals = s["0_n"][used[u_index]]
        q = unique
        regional = np.exp(
            -(((q[:, 1] + 0.065) / 0.030) ** 2)
            - ((q[:, 2] - 0.005) / 0.075) ** 2
            - (q[:, 0] / 0.065) ** 2
        )
        weights = 0.00075 + 0.0025 * regional if group == 0 else np.full(len(q), 0.002)
        local = (
            (abs(q[:, 0]) > 0.065)
            & (q[:, 1] > -0.005)
            & (q[:, 2] > -0.035)
            & (q[:, 2] < 0.050)
        )
        weights = np.where(local, np.maximum(weights, 0.008), weights)
        normals = normals * weights[:, None]
        prefix = OUT / f"build-{group}"
        unique.astype("<f4").tofile(str(prefix) + "-v.bin")
        f.astype("<u4").tofile(str(prefix) + "-f.bin")
        normals.astype("<f4").tofile(str(prefix) + "-n.bin")
        target = target_faces if group == 0 else 1400 if group in (2, 5) else 500
        job = {
            "positions": str(prefix) + "-v.bin",
            "faces": str(prefix) + "-f.bin",
            "normals": str(prefix) + "-n.bin",
            "target": target,
            "error": 0.002 if group == 0 else 0.001,
            "output": str(prefix) + "-result.bin",
        }
        Path(str(prefix) + ".json").write_text(json.dumps(job), encoding="utf-8")
        result = subprocess.run(
            ["node", str(ROOT / "tools/simplify.mjs"), str(prefix) + ".json"],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr[:2000])
        print(result.stdout.strip(), flush=True)
        nf = np.fromfile(job["output"], dtype="<u4").reshape(-1, 3)
        retained = np.unique(nf)
        nv = unique[retained]
        nf = np.searchsorted(retained, nf)
        tree = cKDTree(v)
        distance, closest = tree.query(nv)
        ids = used[closest]
        outer = shaped[ids]
        offset = sum(len(x) for x in points)
        points.append(outer)
        triangles.append(nf + offset)
        dermal = (s["0_epidermis_mm"][ids] + s["0_dermis_mm"][ids]) * 0.001
        fat = s["0_superficial_fat_mm"][ids] * 0.001

        depths.append(np.clip(dermal + fat, 0.0005, 0.012))
        dermals.append(dermal)
        groups.append(np.full(len(nv), group))
        report.append(
            {
                "component": int(group),
                "nodes": len(nv),
                "faces": len(nf),
                "authored_vertex_distance_mm": float(distance.max() * 1000),
            }
        )
        print(json.dumps(report[-1]), flush=True)
    outer = np.vstack(points)
    faces = np.vstack(triangles).astype(np.uint32)
    N = len(outer)
    group = np.concatenate(groups)
    authored_outer = outer.copy()
    # Repair only tiny folded reduction wedges; dense skin stays unchanged and
    # is bound with its exact rest offset. This is an explicit cage approximation.
    edges = np.unique(
        np.sort(
            np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        ),
        axis=0,
    )
    graph = coo_matrix(
        (
            np.ones(len(edges) * 2),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(N, N),
    ).tocsr()
    degree = np.asarray(graph.sum(1)).ravel()
    repair_history = []
    for repair in range(60):
        fn = np.cross(
            outer[faces[:, 1]] - outer[faces[:, 0]],
            outer[faces[:, 2]] - outer[faces[:, 0]],
        )
        fn /= np.linalg.norm(fn, axis=1, keepdims=True)
        rn = trimesh.Trimesh(outer, faces, process=False).vertex_normals.copy()
        dots = np.einsum("fi,fki->fk", fn, rn[faces])
        badfaces = (dots < 0.005).any(1)
        repair_history.append(int(badfaces.sum()))
        if not badfaces.any():
            break
        badnodes = np.unique(faces[badfaces])
        mean = graph @ outer / degree[:, None]
        proposed = outer[badnodes] + 0.25 * (mean[badnodes] - outer[badnodes])
        delta = proposed - authored_outer[badnodes]
        length = np.linalg.norm(delta, axis=1)
        delta *= np.minimum(1.0, 0.0003 / np.maximum(length, 1e-15))[:, None]
        outer[badnodes] = authored_outer[badnodes] + delta
    print(
        json.dumps(
            {
                "cage_wedge_repairs": repair_history,
                "maximum_cage_shift_mm": float(
                    np.linalg.norm(outer - authored_outer, axis=1).max() * 1000
                ),
            }
        ),
        flush=True,
    )
    normal = trimesh.Trimesh(outer, faces, process=False).vertex_normals.copy()
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    fn = np.cross(
        outer[faces[:, 1]] - outer[faces[:, 0]], outer[faces[:, 2]] - outer[faces[:, 0]]
    )
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    dots = np.einsum("fi,fki->fk", fn, normal[faces])
    problem = np.unique(faces[dots.min(1) < 0.01])
    flat = faces.ravel()
    incidence = coo_matrix(
        (np.ones(len(flat)), (flat, np.repeat(np.arange(len(faces)), 3))),
        shape=(N, len(faces)),
    ).tocsr()
    cone_failed = []
    for n in problem:
        rows = incidence.indices[incidence.indptr[n] : incidence.indptr[n + 1]]
        planes = fn[rows]
        lp = linprog(
            [0, 0, 0, -1],
            A_ub=np.c_[-planes, np.ones(len(planes))],
            b_ub=np.zeros(len(planes)),
            bounds=[(-1, 1)] * 3 + [(0, 1)],
            method="highs",
        )
        if lp.success and lp.x[3] > 1e-5:
            normal[n] = lp.x[:3] / np.linalg.norm(lp.x[:3])
        else:
            cone_failed.append(int(n))
    print(
        json.dumps(
            {
                "normal_cone_repaired": len(problem) - len(cone_failed),
                "normal_cone_failed": len(cone_failed),
            }
        ),
        flush=True,
    )
    if cone_failed:
        failed = cone_failed.copy()
        for repair_pass in range(12):
            selected = set(failed)
            if repair_pass > 0:
                for n in failed:
                    selected.update(
                        graph.indices[graph.indptr[n] : graph.indptr[n + 1]].tolist()
                    )
            selected = np.array(sorted(selected))
            mean = graph @ outer / degree[:, None]
            outer[selected] = 0.4 * outer[selected] + 0.6 * mean[selected]
            normal = trimesh.Trimesh(outer, faces, process=False).vertex_normals.copy()
            fn = np.cross(
                outer[faces[:, 1]] - outer[faces[:, 0]],
                outer[faces[:, 2]] - outer[faces[:, 0]],
            )
            fn /= np.linalg.norm(fn, axis=1, keepdims=True)
            dots = np.einsum("fi,fki->fk", fn, normal[faces])
            problem = np.unique(faces[dots.min(1) < 0.01])
            failed = []
            for n in problem:
                rows = incidence.indices[incidence.indptr[n] : incidence.indptr[n + 1]]
                planes = fn[rows]
                lp = linprog(
                    [0, 0, 0, -1],
                    A_ub=np.c_[-planes, np.ones(len(planes))],
                    b_ub=np.zeros(len(planes)),
                    bounds=[(-1, 1)] * 3 + [(0, 1)],
                    method="highs",
                )
                if lp.success and lp.x[3] > 1e-5:
                    normal[n] = lp.x[:3] / np.linalg.norm(lp.x[:3])
                else:
                    failed.append(int(n))
            print(
                json.dumps(
                    {
                        "pinna_local_repair_pass": repair_pass,
                        "remaining_impossible_normal_cones": len(failed),
                    }
                ),
                flush=True,
            )
            if not failed:
                break
        if failed:
            (OUT / "cone-failure.json").write_text(
                json.dumps(
                    {"nodes": failed, "positions_gltf_m": outer[failed].tolist()},
                    indent=2,
                ),
                encoding="utf-8",
            )
            raise ValueError(
                "Coarse surface wedge cannot admit a physical layered volume; see cone-failure.json"
            )
    depth = np.concatenate(depths)
    skin = np.concatenate(dermals)
    original_depth = depth.copy()
    sorted_f = np.sort(faces, axis=1)
    aa, bb, cc = sorted_f.T
    layer_nodes = []
    for l in range(2):
        a, b, c = aa + l * N, bb + l * N, cc + l * N
        A, B, C = a + N, b + N, c + N
        layer_nodes.append(
            np.stack(
                (np.c_[a, b, c, A], np.c_[b, c, A, B], np.c_[c, A, B, C]), 1
            ).reshape(-1, 4)
        )
    nodes = np.vstack(layer_nodes).astype(np.uint32)
    layers = np.repeat(np.arange(2, dtype=np.uint32), len(faces) * 3)
    quality_repairs = []
    from untangle_layered_volume import untangle

    # Cell orientation belongs to the original reference complex. Recomputing
    # it from optimized outer faces can reject an orientation-preserving repair
    # merely because a face rotated relative to its old extrusion direction.
    reference = np.sign(
        np.einsum(
            "ij,ij->i",
            np.cross(outer[bb] - outer[aa], outer[cc] - outer[aa]),
            -normal[aa],
        )
    )
    expected = np.tile(np.repeat(reference, 3), 2)
    rest, repair = untangle(outer, faces, normal, depth, skin, nodes, OUT)
    outer = rest[:N]
    p = rest[nodes]
    D = np.swapaxes(p[:, 1:] - p[:, :1], 1, 2)
    det = np.linalg.det(D)
    quality_repairs = repair["history"]
    history = []
    # Reference orientation follows the consistently extruded face-normal prism.
    depth = np.einsum("ij,ij->i", outer - rest[2 * N :], normal)
    middle_gap = np.einsum("ij,ij->i", outer - rest[N : 2 * N], normal)
    condition = np.linalg.cond(D)
    if (
        (det * expected <= 0).any()
        or depth.min() < 0.00025
        or middle_gap.min() < 0.000125
        or (depth - middle_gap).min() < 0.000125
        or condition.max() > 10000
    ):
        raise ValueError(
            "Rest-volume untangling did not meet the thickness/conditioning gate"
        )
    negative = det < 0
    nodes[negative, 1], nodes[negative, 2] = (
        nodes[negative, 2].copy(),
        nodes[negative, 1].copy(),
    )
    p = rest[nodes]
    D = np.swapaxes(p[:, 1:] - p[:, :1], 1, 2)
    V = np.linalg.det(D) / 6
    inverse = np.linalg.inv(D)
    gradient = np.zeros((len(nodes), 4, 3))
    gradient[:, 1:] = inverse
    gradient[:, 0] = -inverse.sum(1)
    E = np.where(layers == 0, 120000.0, 18000.0)
    nu = np.where(layers == 0, 0.45, 0.48)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)) + mu
    mass = np.zeros(3 * N)
    np.add.at(
        mass, nodes.ravel(), np.repeat(V * np.where(layers == 0, 1050, 950) / 4, 4)
    )
    arrays = {
        "rest": rest,
        "nodes": nodes,
        "gradients": gradient,
        "material": np.c_[V, mu, lam],
        "mass": mass,
        "faces": faces,
        "layers": layers,
        "groups": group.astype(np.uint32),
        "normals": normal,
    }
    assets = {}
    for name, a in arrays.items():
        a = a.astype("<u4" if a.dtype.kind in "iu" else "<f8")
        path = OUT / (name + ".bin")
        a.tofile(path)
        assets[name] = {
            "url": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path),
            "count": a.size,
        }
    construction = {
        "components": report,
        "bounds_m": [outer.min(0).tolist(), outer.max(0).tolist()],
        "positive_cells": bool((V > 0).all()),
        "minimum_rest_volume_m3": float(V.min()),
        "maximum_condition": float(np.linalg.cond(D).max()),
        "effective_depth_mm": np.quantile(
            depth * 1000, [0, 0.01, 0.5, 0.99, 1]
        ).tolist(),
        "curvature_reduced_nodes": int((depth < original_depth).sum()),
        "reduction_history": history,
        "cage_wedge_repairs": repair_history,
        "quality_repairs": quality_repairs,
        "maximum_cage_shift_mm": float(
            np.linalg.norm(outer - authored_outer, axis=1).max() * 1000
        ),
        "full_head_surface_reduction": True,
        "anatomical_contact_accepted": False,
        "elapsed_s": time.perf_counter() - start,
    }
    model = {
        "schema": "cybr-layered-fem-v1",
        "surface_nodes": N,
        "nodes": 3 * N,
        "tetrahedra": len(nodes),
        "assets": assets,
        "units": "meters",
        "coordinate_frame": "gltf",
        "source_skin_sha256": sha(source),
        "skin_shape_url": shape_file.relative_to(ROOT).as_posix(),
        "full_head": True,
        "construction": construction,
        "limitations": [
            "Full outer scalp/face/pinna coverage; oral/ocular linings follow the cage but have no dedicated thin-feature tetrahedral cells.",
            "Prescribed inner substrate, not accepted anatomical skull/muscle contact.",
            "Experimental homogenized skin and fat parameters.",
        ],
    }
    (OUT / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "cage.npz", v=outer, f=faces, n=normal, groups=group)
    print(
        json.dumps({"surface_nodes": N, "cells": len(nodes), **construction}),
        flush=True,
    )
