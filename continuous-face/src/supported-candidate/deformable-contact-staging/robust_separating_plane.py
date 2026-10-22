"""Proof-only finite-triangle separating axes; never contact derivatives.

Axis construction is deliberately not trusted. Long-double SAT candidates are
rounded to binary64 and treated as arbitrary, exact represented directions.
Only outward-rounded projection of ALL nine vertex differences can establish
separation. A failed proof means unresolved, never nonintersection.
"""
import numpy as np
from moving_triangle_contact import _down, _up, _imul, _iadd

EDGES = ((0, 1), (1, 2), (2, 0))


def projection_lower(P, Q, axes):
    """Lower bounds on every (A vertex - B vertex) dot axis.

    P,Q are one triangle's coordinate intervals, each shape (3,3).
    Returned shape is (number of axes,). Nonfinite arithmetic cannot certify.
    """
    axes = np.asarray(axes, float).reshape(-1, 3)
    lo = _down(P[0][:, None, :] - Q[1][None, :, :])
    hi = _up(P[1][:, None, :] - Q[0][None, :, :])
    product = _imul((lo[None], hi[None]),
                    (axes[:, None, None, :], axes[:, None, None, :]))
    total = (np.zeros(product[0].shape[:-1]), np.zeros(product[0].shape[:-1]))
    for j in range(3):
        total = _iadd(total, (product[0][..., j], product[1][..., j]))
    out = total[0].min((1, 2))
    return np.where(np.isfinite(out), out, -np.inf)


def candidate_axes(A, B):
    """Finite SAT axes, including in-plane axes for coplanar triangles.

    Completeness of this list and accuracy of construction are NOT assumptions
    in acceptance. Approximate or omitted axes can only cause false rejection.
    Scaling by max component avoids overflow and unnecessary square roots.
    """
    a = np.asarray(A, np.longdouble)
    b = np.asarray(B, np.longdouble)
    ea = np.array([a[j] - a[i] for i, j in EDGES])
    eb = np.array([b[j] - b[i] for i, j in EDGES])
    na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
    raw = [na, nb]
    raw += [np.cross(e, f) for e in ea for f in eb]
    raw += [np.cross(na, e) for e in ea]
    raw += [np.cross(nb, e) for e in eb]
    axes = []
    for v in raw:
        scale = np.max(np.abs(v))
        if not np.isfinite(scale) or scale <= 0:
            continue
        n = np.asarray(v / scale, float)
        if np.isfinite(n).all() and np.any(n):
            axes.extend((n, -n))
    return np.asarray(axes, float).reshape(-1, 3)


def plane_certificate(A, B, *, tol, endpoints=()):
    """Prove finite separation above the ORIGINAL ambiguity floor.

    tol is the existing 256*eps*coordinate-scale floor, not a new tolerance.
    Axes are not unit-length. An outward upper L1 norm bounds their L2 norm,
    so lower > upper(tol * L1) proves true Euclidean clearance > tol.
    For an affine segment, the SAME axis must strictly separate both interval
    endpoint pairs. Their convex combinations then remain separated throughout.
    Neither endpoint samples nor an exact midpoint alone can accept a path.
    """
    axes = candidate_axes(A, B)
    if not len(axes) or not np.isfinite(tol) or tol < 0:
        return None
    lower = projection_lower((A, A), (B, B), axes)
    norm_upper = np.zeros(len(axes))
    for j in range(3):
        norm_upper = _up(norm_upper + np.abs(axes[:, j]))
    floor_upper = _up(tol * norm_upper)
    valid = np.isfinite(norm_upper) & (lower > floor_upper)
    score = lower.copy()
    endpoint_lowers = []
    for P, Q in endpoints:
        ep = projection_lower(P, Q, axes)
        endpoint_lowers.append(ep)
        valid &= ep > 0
        score = np.minimum(score, ep)
    if not valid.any():
        return None
    # Rank only; this approximate division has no role in the proof.
    score = np.where(valid, score / norm_upper, -np.inf)
    k = int(np.argmax(score))
    return dict(normal=axes[k].copy(), lower=float(lower[k]),
                floor_upper=float(floor_upper[k]), norm_upper=float(norm_upper[k]),
                endpoint_lowers=[float(v[k]) for v in endpoint_lowers],
                guarantee='outward-rounded all-nine vertex differences')
