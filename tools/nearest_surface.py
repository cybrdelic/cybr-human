"""Exact closest-triangle query shared by the tissue builder."""

import numpy as np
import trimesh


def nearest_faces(points, triangles, tree, centers, radii):
    """Exact finite nearest faces, certified by centroid/radius lower bounds."""
    _, first = tree.query(points, k=8)
    candidates = triangles[first]
    cp = trimesh.triangles.closest_point(
        candidates.reshape(-1, 3, 3), np.repeat(points, 8, axis=0)
    ).reshape(-1, 8, 3)
    best = np.linalg.norm(cp - points[:, None], axis=2).min(1)
    groups = tree.query_ball_point(points, best + radii.max() + 1e-12)
    ids = np.concatenate([np.sort(g) for g in groups])
    owners = np.repeat(np.arange(len(points)), [len(g) for g in groups])
    lower = np.linalg.norm(centers[ids] - points[owners], axis=1) - radii[ids]
    keep = lower <= best[owners] + 1e-12
    ids, owners = ids[keep], owners[keep]
    cp = trimesh.triangles.closest_point(triangles[ids], points[owners])
    dist2 = np.einsum("ij,ij->i", cp - points[owners], cp - points[owners])
    minimum = np.full(len(points), np.inf)
    np.minimum.at(minimum, owners, dist2)
    winner = np.full(len(points), len(ids), np.int64)
    match = dist2 == minimum[owners]
    np.minimum.at(winner, owners[match], np.flatnonzero(match))
    if np.any(winner == len(ids)):
        raise ValueError("Finite embedding omitted a source point")
    return ids[winner], cp[winner], np.sqrt(minimum)
