"""Independent checks on the actual immutable GLB, not builder arrays."""

import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
pointer = (
    ROOT
    / "output"
    / (
        "runtime-candidate.json"
        if os.environ.get("TISSUE_CANDIDATE") == "1"
        else "runtime-current.json"
    )
)
runtime = json.loads(pointer.read_text(encoding="utf-8"))
assert runtime["build"] == os.environ.get("TISSUE_BUILD", runtime["build"])
model = json.loads(
    (ROOT / runtime["tissueBase"].lstrip("/") / "model.json").read_text(
        encoding="utf-8"
    )
)
if not model.get("registration_url"):
    print("No anatomical foundation in this historical build")
    raise SystemExit(0)
scene = trimesh.load(ROOT / model["skin_url"].lstrip("/"), force="scene")
# Verify the actual ocular export, including winding and the exact iris seam.
# A centroid-clipped globe can pass the skin topology test while leaving a
# jagged limbus or rendering the back of the globe through the opening.
ocular = []
for name, globe in scene.geometry.items():
    if not name.endswith("_sclera"):
        continue
    gv, gf = globe.vertices, globe.faces
    sphere = np.linalg.lstsq(
        np.c_[2 * gv, np.ones(len(gv))], (gv * gv).sum(1), rcond=None
    )[0]
    centre = sphere[:3]
    triangles = gv[gf]
    orientation = np.einsum(
        "ij,ij->i",
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        triangles.mean(1) - centre,
    )
    assert orientation.min() > 0, "Inward ocular faces"
    seam = gv[np.abs(gv[:, 2] - gv[:, 2].max()) < 1e-8]
    radial = np.linalg.norm(seam[:, :2] - centre[:2], axis=1)
    assert np.max(abs(radial - 0.00572)) < 1e-7, "Jagged sclera/iris seam"
    ocular.append(
        {
            "name": name,
            "minimum_outward_orientation": float(orientation.min()),
            "limbus_radius_error_m": float(np.max(abs(radial - 0.00572))),
        }
    )
skin = next(
    mesh
    for name, mesh in scene.geometry.items()
    if "Continuous_anatomical_skin" in name
)
skin = trimesh.Trimesh(skin.vertices, skin.faces, process=True)
v = skin.vertices
f = skin.faces
assert np.isfinite(v).all()
assert len(np.unique(np.sort(f, 1), axis=0)) == len(f), "Duplicate skin faces"
edges, count = np.unique(
    np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
    axis=0,
    return_counts=True,
)
assert count.max() <= 2, "Nonmanifold skin edge"
boundary = edges[count == 1]
used = np.unique(boundary)
local = np.searchsorted(used, boundary)
graph = coo_matrix(
    (
        np.ones(len(local) * 2),
        (np.r_[local[:, 0], local[:, 1]], np.r_[local[:, 1], local[:, 0]]),
    ),
    shape=(len(used), len(used)),
).tocsr()
assert np.all(np.asarray(graph.sum(1)).ravel() == 2), (
    "Open or branching aperture boundary"
)
loops = connected_components(graph, directed=False)[0]
assert loops == 6, f"Expected five facial apertures and one neck boundary, got {loops}"
centres = v[f].mean(1)
cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
frontal = (
    (abs(centres[:, 0]) < 0.055)
    & (centres[:, 1] > -0.060)
    & (centres[:, 1] < 0.050)
    & (centres[:, 2] > 0.045)
)
folds = int((cross[frontal, 2] < -1e-14).sum())
assert folds == 0, f"Folded frontal chart triangles: {folds}"
registration = json.loads(
    (ROOT / model["registration_url"].lstrip("/")).read_text(encoding="utf-8")
)
outside = sum(p["outside_vertices"] for p in registration["parts"])
assert outside == 0, "Internal reference vertices outside the authored chart"
result = {
    "build": runtime["build"],
    "vertices": len(v),
    "triangles": len(f),
    "boundary_loops": loops,
    "folded_frontal_triangles": folds,
    "registered_parts": len(registration["parts"]),
    "chart_outside_vertices": outside,
    "minimum_chart_clearance_mm": min(
        p["minimum_chart_clearance_mm"] for p in registration["parts"]
    ),
    "ocular_seams": ocular,
    "limitations": "Chart containment is a sampled construction check, not global triangle contact or anatomical validation.",
}
(ROOT / "output/verification").mkdir(parents=True, exist_ok=True)
(ROOT / "output/verification/head-topology.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result))
