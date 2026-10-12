from pathlib import Path
import json, hashlib, struct
import numpy as np, trimesh
from scipy.spatial import cKDTree
from nearest_surface import nearest_faces
from build_neutral_tissue import OUT, ROOT


def asset(name, array, output_dir):
    p = Path(output_dir) / (name + ".bin")
    array.astype("<f8").tofile(p)
    return {
        "url": p.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "count": array.size,
    }


def main(output_dir=OUT):
    OUT = Path(output_dir)
    model = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
    source = np.load(OUT / "skin.npz")
    c = np.load(OUT / "cage.npz")
    v = c["v"]
    f = c["f"]
    # Read the GLB's actual mesh order and validate every source part length.
    raw = (ROOT / model["skin_url"]).read_bytes()
    size = struct.unpack_from("<I", raw, 12)[0]
    g = json.loads(raw[20 : 20 + size])
    names = sorted([n for n in g["nodes"] if "mesh" in n], key=lambda n: n["mesh"])
    shape = json.loads((OUT / "shape.json").read_text(encoding="utf-8"))
    for i, node in enumerate(names):
        assert g["accessors"][
            g["meshes"][node["mesh"]]["primitives"][0]["attributes"]["POSITION"]
        ]["count"] == len(source[f"{i}_v"])
        shape["parts"][i]["node"] = node["name"]
    (OUT / "shape.json").write_text(json.dumps(shape), encoding="utf-8")
    model["skin_main_node"] = names[0]["name"]
    triangle = v[f]
    areas = (
        np.linalg.norm(
            np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]),
            axis=1,
        )
        / 2
    )
    edge_map = {}
    for j, tri in enumerate(f):
        for a, b, o in [
            (tri[0], tri[1], tri[2]),
            (tri[1], tri[2], tri[0]),
            (tri[2], tri[0], tri[1]),
        ]:
            edge_map.setdefault(tuple(sorted((int(a), int(b)))), []).append(
                (int(a), int(b), int(o), j)
            )
    hinges = []
    D = 120000 * 0.0012**3 / (12 * (1 - 0.45**2))
    for edge, adj in edge_map.items():
        if len(adj) != 2:
            continue
        a, b, o, j = adj[0]
        _, _, p, k = adj[1]
        e = v[b] - v[a]
        n1 = np.cross(e, v[o] - v[a])
        n2 = np.cross(v[p] - v[a], e)
        theta = np.arctan2(
            np.dot(np.cross(n1, n2), e) / np.linalg.norm(e), np.dot(n1, n2)
        )
        stiffness = D * np.dot(e, e) / (areas[j] + areas[k])
        hinges.append([a, b, o, p, theta, stiffness])
    model["assets"]["hinges"] = asset("hinges", np.asarray(hinges), OUT)
    model["shell_hinges"] = len(hinges)
    model["shell_description"] = {
        "stretching": "Existing outer dermal solid FEM",
        "bending": "Rest-dihedral energy with exact forces and SPD Gauss-Newton search metric",
        "effective_thickness_m": 0.0012,
        "modulus_pa": 120000,
        "poisson_ratio": 0.45,
        "micro_wrinkles_resolved": False,
    }
    # Growth is restricted to the adipose cells; mass increases with grown volume.
    layers = np.fromfile(ROOT / model["assets"]["layers"]["url"], dtype="<u4")
    ids = np.fromfile(ROOT / model["assets"]["nodes"]["url"], dtype="<u4").reshape(
        -1, 4
    )
    rest = np.fromfile(ROOT / model["assets"]["rest"]["url"], dtype="<f8").reshape(
        -1, 3
    )
    p = rest[ids].mean(1)
    x, z, y = p.T * np.array([1000, 1000, -1000])[:, None]
    # The slider means actual adipose volume growth. Regional thickness is
    # already represented by the volume; multiplying by it again attenuated
    # the control and hid changes in the thinner regions.
    weight = (layers == 1).astype(float)
    pinna = (abs(x) > 67) & (y > 0) & (z > -30) & (z < 45)
    nasal = (abs(x) < 18) & (y < -50) & (z > -24) & (z < 45)
    vermilion = (abs(x) < 26) & (y < -50) & (abs(z + 35.5) < 7)
    lids = np.zeros(len(x), bool)
    for side in [-1, 1]:
        lids |= (((x - side * 31) / 18) ** 2 + ((z - 30) / 12) ** 2 < 1) & (y < -40)
    weight[pinna | nasal | vermilion | lids] = 0
    model["adipose_growth_scope"] = (
        "Authored adipose label: excludes pinnae, nasal support, vermilion and eyelid thin features; not segmented anatomical fat compartments."
    )
    model["assets"]["fatWeights"] = asset("fatWeights", weight, OUT)
    # The auricle is attached tissue, not an immovable skull surface. Keep
    # the connected root through the volume, but allow its inner nodes to
    # respond rather than crushing the outer coat against a fixed ear shell.
    nx, nz, ny = rest.T * np.array([1000, 1000, -1000])[:, None]
    auricle = (abs(nx) > 67) & (ny > 0) & (nz > -30) & (nz < 45)
    fixed = (np.arange(model["nodes"]) >= 2 * model["surface_nodes"]).astype(float)
    fixed[auricle] = 0
    fixed[rest[:, 1] < rest[: model["surface_nodes"], 1].min() + 0.001] = 1
    model["assets"]["fixedNodes"] = asset("fixedNodes", fixed, OUT)
    model["boundary_description"] = (
        "Prescribed head substrate and neck base; connected auricular inner nodes remain free. No activated muscles or reciprocal jaw."
    )
    from anatomical_support import contact_planes

    planes, contact_report = contact_planes(rest, (ROOT / model["skin_url"]).parent)
    model["assets"]["contactPlanes"] = asset("contactPlanes", planes, OUT)
    model["anatomical_contact"] = contact_report
    parts = []
    face_ids = []
    barys = []
    offset = 0
    reports = []
    centers = triangle.mean(1)
    radii = np.linalg.norm(triangle - centers[:, None], axis=2).max(1)
    tree = cKDTree(centers)
    for i, node in enumerate(names):
        points = source[f"{i}_v"][:, [0, 2, 1]] * [1, 1, -1]
        count = len(points)
        fid = np.full(count, 0xFFFFFFFF, np.uint32)
        weights = np.zeros((count, 2), np.float32)
        active = not any(
            term in node["name"].lower()
            for term in ["sclera", "iris", "pupil", "conjunctival"]
        )
        worst = 0
        if active:
            for begin in range(0, count, 512):
                ids, cp, distance = nearest_faces(
                    points[begin : begin + 512], triangle, tree, centers, radii
                )
                w = trimesh.triangles.points_to_barycentric(triangle[ids], cp)
                w = np.clip(w, 0, 1)
                w /= w.sum(1)[:, None]
                fid[begin : begin + len(ids)] = ids
                weights[begin : begin + len(ids)] = w[:, :2]
                worst = max(worst, float(distance.max()))
        parts.append(
            {
                "index": i,
                "vertex_count": count,
                "embedding": {"offset": offset, "count": count},
                "active_vertex_count": int((fid != 0xFFFFFFFF).sum()),
            }
        )
        reports.append(
            {"part": i, "node": node["name"], "maximum_offset_mm": worst * 1000}
        )
        face_ids.append(fid)
        barys.append(weights)
        offset += count
    arrays = {
        "faces": f.astype("<u4"),
        "vertex_faces": np.concatenate(face_ids).astype("<u4"),
        "barycentric2": np.concatenate(barys).astype("<f4"),
    }
    desc = {}
    cursor = 0
    with (OUT / "surface-embedding.bin").open("wb") as file:
        for name, a in arrays.items():
            raw = a.tobytes()
            file.write(raw)
            desc[name] = {
                "byte_offset": cursor,
                "byte_length": len(raw),
                "length": a.size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            cursor += len(raw)
    (OUT / "surface-embedding.json").write_text(
        json.dumps(
            {
                "schema": "cybr-full-head-surface-embedding-v1",
                "parts": parts,
                **desc,
                "reports": reports,
            }
        ),
        encoding="utf-8",
    )
    (OUT / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    from build_tissue_transfer import main as build_transfer

    build_transfer(output_dir=OUT)
    print(
        json.dumps(
            {
                "volume_cells": model["tetrahedra"],
                "skin_bending_hinges": len(hinges),
                "surface_vertices": offset,
                "main_embedding_offset_mm": reports[0]["maximum_offset_mm"],
                "micro_wrinkles_resolved": False,
            }
        )
    )


if __name__ == "__main__":
    main()
