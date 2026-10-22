"""Proof-only axes for the original disjoint convex VF and EE features.

Neither candidate construction nor approximate closest points are trusted.
Every accepted axis must separate every pair of feature vertices under outward
arithmetic, at both interval endpoints. No complete shared triangle is tested
for strict separation; those may legally meet on their owned shared feature.
"""
import numpy as np
from robust_separating_plane import projection_lower
from moving_triangle_contact import _up


def candidate_feature_axes(A, B):
    a, b = np.asarray(A, np.longdouble), np.asarray(B, np.longdouble)
    raw = []
    if a.shape == (1, 3) and b.shape == (3, 3):
        edges = [b[1]-b[0], b[2]-b[1], b[0]-b[2]]
        normal = np.cross(edges[0], -edges[2])
        raw.append(normal)
        raw.extend(np.cross(normal, e) for e in edges)
    elif a.shape == (2, 3) and b.shape == (2, 3):
        ea, eb = a[1]-a[0], b[1]-b[0]
        normal = np.cross(ea, eb)
        raw.extend((normal, np.cross(normal, ea), np.cross(normal, eb), ea, eb))
        # Parallel segments may need a transverse direction instead of their
        # vanishing cross product. These are still only arbitrary candidates.
        for delta in (x-y for x in a for y in b):
            raw.extend((np.cross(ea, np.cross(delta, ea)),
                        np.cross(eb, np.cross(delta, eb))))
    else:
        raise ValueError('Only disjoint VF or EE convex primitive shapes supported')
    raw.extend(x-y for x in a for y in b)
    axes = []
    for v in raw:
        scale = np.max(np.abs(v))
        if not np.isfinite(scale) or scale <= 0:
            continue
        n = np.asarray(v/scale, float)
        if np.isfinite(n).all() and np.any(n):
            axes.extend((n, -n))
    return np.asarray(axes, float).reshape(-1, 3)


def feature_plane_certificate(A, B, *, tol, endpoints=()):
    """Strictly above unchanged ambiguity floor at midpoint; strict endpoints.

    tol is supplied by the caller from the ORIGINAL global coordinate scale.
    Any accepted n has lower > outward(tol*L1(n)) >= tol*L2(n).
    The same axis must prove strictly positive bounds at both endpoints.
    """
    axes = candidate_feature_axes(A, B)
    if not len(axes) or not np.isfinite(tol) or tol < 0:
        return None
    lower = projection_lower((A, A), (B, B), axes)
    norm_upper = np.zeros(len(axes))
    for j in range(3):
        norm_upper = _up(norm_upper + np.abs(axes[:, j]))
    floor_upper = _up(tol*norm_upper)
    valid = np.isfinite(norm_upper) & (lower > floor_upper)
    score = lower.copy(); endpoint_lowers = []
    for P, Q in endpoints:
        ep = projection_lower(P, Q, axes)
        valid &= ep > 0
        score = np.minimum(score, ep); endpoint_lowers.append(ep)
    if not valid.any():
        return None
    k = int(np.argmax(np.where(valid, score/norm_upper, -np.inf)))
    return dict(normal=axes[k].copy(), lower=float(lower[k]),
                floor_upper=float(floor_upper[k]), norm_upper=float(norm_upper[k]),
                endpoint_lowers=[float(x[k]) for x in endpoint_lowers],
                guarantee='outward-rounded all convex-feature vertex differences')
