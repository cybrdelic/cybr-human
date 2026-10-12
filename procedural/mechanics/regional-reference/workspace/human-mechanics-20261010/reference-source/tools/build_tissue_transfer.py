"""Compact partition-of-unity transfer of solved nodal affine fields.

The ninth-nearest distance sets the support radius: weights vanish at a
support swap. This preserves affine motions exactly without changing FEM.
This spatial transfer is not a substitute for anatomical contact.
"""

from pathlib import Path
import hashlib, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/neutral-tissue/v6"


def main(output_dir=OUT):
    OUT = Path(output_dir)
    model = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
    source = np.load(OUT / "skin.npz")
    rest = np.fromfile(ROOT / model["assets"]["rest"]["url"], dtype="<f8").reshape(
        -1, 3
    )
    tree = cKDTree(rest[: model["surface_nodes"]])
    bundle = json.loads((OUT / "surface-embedding.json").read_text(encoding="utf-8"))
    faces = np.fromfile(ROOT / model["assets"]["faces"]["url"], dtype="<u4").reshape(
        -1, 3
    )
    edges = np.unique(
        np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
        ),
        axis=0,
    )
    length = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    S = model["surface_nodes"]
    graph = coo_matrix(
        (
            np.r_[length, length],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(S, S),
    ).tocsr()
    intrinsic = dijkstra(graph, directed=False, limit=0.08)
    raw = (OUT / "surface-embedding.bin").read_bytes()
    record = bundle["vertex_faces"]
    bound_faces = np.frombuffer(
        raw, dtype="<u4", count=record["length"], offset=record["byte_offset"]
    )
    topology_rejections = 0
    indices, weights, checks = [], [], []
    A = np.array([[0.96, -0.24, 0.03], [0.24, 0.96, 0.04], [0.01, 0, 1.02]])
    translation = np.array([0.003, -0.002, 0.001])
    for part in bundle["parts"]:
        points = source[f"{part['index']}_v"][:, [0, 2, 1]] * [1, 1, -1]
        # Euclidean neighbors across a closed lip, eyelid or folded pinna must
        # not share motion. Reject shortcuts that require a long path on the
        # actual surface graph, anchored to the bound surface triangle.
        distances, candidates = tree.query(points, k=64)
        fid = bound_faces[
            part["embedding"]["offset"] : part["embedding"]["offset"] + len(points)
        ]
        if part["active_vertex_count"]:
            anchor = faces[fid]
            path = np.minimum.reduce(
                [intrinsic[anchor[:, i, None], candidates] for i in range(3)]
            )
            face_radius = np.linalg.norm(rest[anchor] - points[:, None], axis=2).max(1)
            allowed = path <= np.maximum(0.004, 2.5 * distances + face_radius[:, None])
            ranked = np.where(allowed, distances, np.inf)
            ordering = np.argsort(ranked, axis=1)[:, :9]
            d = np.take_along_axis(ranked, ordering, axis=1)
            ids = np.take_along_axis(candidates, ordering, axis=1)
            if not np.isfinite(d).all():
                raise ValueError(
                    "Insufficient local topology support; refine the cage instead of crossing a feature"
                )
            topology_rejections += int((~allowed[:, :9]).sum())
        else:
            d, ids = distances[:, :9], candidates[:, :9]
        r = np.maximum(d[:, 8:9], 1e-9)
        s = np.clip(d[:, :8] / r, 0, 1)
        w = (1 - s) ** 4 * (4 * s + 1)
        w /= np.maximum(w.sum(1, keepdims=True), 1e-30)
        indices.append(ids[:, :8].astype("<f4"))
        weights.append(w.astype("<f4"))
        if part["active_vertex_count"]:
            nodes = rest[ids[:, :8]]
            posed = nodes @ A.T + translation
            reproduced = (
                w[:, :, None] * (posed + (points[:, None] - nodes) @ A.T)
            ).sum(1)
            error = float(np.max(np.abs(reproduced - (points @ A.T + translation))))
            assert error < 1e-12
            checks.append(
                {
                    "part": part["index"],
                    "tested_vertices": len(points),
                    "affine_error_m": error,
                }
            )
    data = np.concatenate(
        [np.concatenate(indices), np.concatenate(weights)], axis=1
    ).astype("<f4")
    path = OUT / "smooth-transfer.bin"
    data.tofile(path)
    bundle["smooth_transfer"] = {
        "url": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nodes_per_vertex": 8,
        "stride_floats": 16,
        "description": "Normalized compact Wendland affine fields with surface-graph shortcut rejection; no contact guarantee",
    }
    (OUT / "surface-embedding.json").write_text(json.dumps(bundle), encoding="utf-8")
    report = {
        "vertices": len(data),
        "tested_active_vertices": sum(c["tested_vertices"] for c in checks),
        "support_nodes": 8,
        "affine_reproduction_error_m": max(c["affine_error_m"] for c in checks),
        "rejected_shortcut_neighbors": topology_rejections,
        "parts": checks,
    }
    (OUT / "transfer-check.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
