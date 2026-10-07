"""Exact nearest internal triangle anchors in the shared atlas frame.

These are local, fixed contact tangents. They are not a global collision
detector, sliding fascia model, muscle activation or articulated jaw solver.
"""

import json
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent


def internal_mesh(head_dir, kinds=("bone", "cartilage")):
    report = json.loads((Path(head_dir) / "report.json").read_text(encoding="utf-8"))
    source = np.load(ROOT / "assets/anatomy/geometry.npz")
    vertices = []
    faces = []
    parts = []
    offset = 0
    for p in report["construction"]["parts"]:
        if p["kind"] not in kinds:
            continue
        i = p["part"]
        v = source[f"{i}_v"].astype(float)
        vertices.append(v[:, [0, 2, 1]] * [1, 1, -1])
        faces.append(source[f"{i}_f"] + offset)
        parts.extend([i] * len(source[f"{i}_f"]))
        offset += len(v)
    return trimesh.Trimesh(
        np.vstack(vertices), np.vstack(faces), process=False
    ), np.asarray(parts)


def closest(points, mesh):
    anchors = []
    distances = []
    faces = []
    for start in range(0, len(points), 128):
        p, d, f = trimesh.proximity.closest_point(mesh, points[start : start + 128])
        anchors.append(p)
        distances.append(d)
        faces.append(f)
    return np.vstack(anchors), np.concatenate(distances), np.concatenate(faces)


def contact_planes(rest, head_dir):
    mesh, parts = internal_mesh(head_dir)
    point, distance, ids = closest(rest, mesh)
    direction = rest - point
    # Cropped cortical parts are not one closed solid. Face-normal signs at
    # an open edge cannot establish inside/outside. These tangents protect an
    # exterior starting point locally; do not claim a signed-distance field.
    normals = direction / np.maximum(distance[:, None], 1e-12)
    return np.c_[point, normals, np.full(len(rest), 0.00001)], {
        "method": "nearest permitted bone/cartilage triangle; fixed local Euclidean-distance tangent",
        "frame": "gltf metres, native atlas registration",
        "triangles": len(mesh.faces),
        "queries": len(rest),
        "minimum_anchor_distance_mm": float(distance.min() * 1000),
        "signed_inside_test": False,
        "part_count": len(np.unique(parts[ids])),
        "global_collision": False,
        "moving_contact": False,
    }
