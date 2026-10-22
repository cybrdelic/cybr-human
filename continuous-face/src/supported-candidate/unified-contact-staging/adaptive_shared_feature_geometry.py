"""Exact-topology shared-triangle predicate with safe interval acceleration.

No geometric distance tolerance is used to excuse an intersection. The only
permitted locus is the explicitly supplied exact common vertex or edge.
Outward-rounded determinant intervals certify easy negative cases. Every other
pair is decided by the independent exact dyadic-integer cone/hinge oracle.
"""
import numpy as np

from exact_shared_feature_geometry import robust_forbidden_intersection


def pack_shared_indices(shared):
    counts = np.fromiter((len(s) for s in shared), dtype=np.intp, count=len(shared))
    if np.any((counts != 1) & (counts != 2)):
        raise ValueError('Each pair must have one or two exact shared vertices')
    indices = np.zeros((len(shared), 2), dtype=np.intp)
    for i, s in enumerate(shared):
        indices[i, :len(s)] = s
        if len(s) == 1:
            indices[i, 1] = s[0]
    indices.flags.writeable = counts.flags.writeable = False
    return indices, counts


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _sub(a, b):
    return _down(a[0] - b[1]), _up(a[1] - b[0])


def _mul(a, b):
    values = np.stack((a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]))
    return _down(values.min(axis=0)), _up(values.max(axis=0))


def _cross(a, b):
    j, k = [1, 2, 0], [2, 0, 1]
    return _sub(_mul((a[0][:, j], a[1][:, j]), (b[0][:, k], b[1][:, k])),
                _mul((a[0][:, k], a[1][:, k]), (b[0][:, j], b[1][:, j])))


def _dot(a, b):
    lo, hi = _mul(a, b)
    low, high = lo[:, 0], hi[:, 0]
    for j in (1, 2):
        low, high = _down(low + lo[:, j]), _up(high + hi[:, j])
    return low, high


def _nonzero(interval):
    return (interval[0] > 0) | (interval[1] < 0)


def _same_strict_sign(a, b):
    return ((a[0] > 0) & (b[0] > 0)) | ((a[1] < 0) & (b[1] < 0))


def robust_forbidden_intersections(A, B, shared_points, shared_counts=None, *, return_diagnostics=False):
    A, B, S = (np.asarray(x, dtype=float) for x in (A, B, shared_points))
    if A.ndim != 3 or A.shape[1:] != (3, 3) or B.shape != A.shape:
        raise ValueError('A and B must have matching shape (n,3,3)')
    n = len(A)
    if S.ndim != 3 or S.shape[0] != n or S.shape[1] not in (1, 2) or S.shape[2] != 3:
        raise ValueError('Explicit shared points must have shape (n,1|2,3)')
    counts = np.full(n, S.shape[1], dtype=int) if shared_counts is None else np.asarray(shared_counts, dtype=int)
    if counts.shape != (n,) or np.any((counts < 1) | (counts > S.shape[1])):
        raise ValueError('Explicit shared counts must contain 1 or 2')
    if not np.isfinite(A).all() or not np.isfinite(B).all() or not np.isfinite(S).all():
        raise ValueError('Finite triangle and shared-point coordinates required')
    edge = counts == 2
    rows = np.arange(n)
    vectors = []
    for T in (A, B):
        same0 = (T == S[:, :1]).all(axis=2)
        if np.any(same0.sum(axis=1) != 1):
            # Degenerate repeated vertices and falsely asserted ownership must
            # not become an exemption. The exact helper fails closed as well.
            raise ValueError('Shared vertex must occur exactly once on each triangle')
        order = np.argsort(same0, axis=1, kind='stable')
        p, q = T[rows, order[:, 0]], T[rows, order[:, 1]]
        if edge.any():
            same1 = (T[edge] == S[edge, 1:2]).all(axis=2)
            if np.any(same1.sum(axis=1) != 1) or np.any((S[edge, 0] == S[edge, 1]).all(axis=1)):
                raise ValueError('Shared edge must consist of two distinct exact triangle vertices')
            other = ~(same0[edge] | same1)
            if np.any(other.sum(axis=1) != 1):
                raise ValueError('Degenerate shared edge triangle')
            p[edge] = S[edge, 1]
            q[edge] = T[rows[edge], np.argmax(other, axis=1)]
        vectors.append((_sub((p, p), (S[:, 0], S[:, 0])),
                        _sub((q, q), (S[:, 0], S[:, 0]))))
    (a, b), (c, d) = vectors
    with np.errstate(invalid='ignore', over='ignore', under='ignore'):
        na, nb = _cross(a, b), _cross(c, d)
        nondegenerate = _nonzero(na).any(axis=1) & _nonzero(nb).any(axis=1)
        # For an exact common edge, distinct planes intersect only on its line.
        # The triangles each contain exactly that edge segment of the line.
        edge_safe = edge & _nonzero(_dot(na, d))
        # For a common vertex S, both opposite vertices strictly on one side
        # of the other plane imply that triangle meets that plane only at S.
        vertex_safe = ~edge & (_same_strict_sign(_dot(na, c), _dot(na, d))
                               | _same_strict_sign(_dot(nb, a), _dot(nb, b)))
        safe = nondegenerate & (edge_safe | vertex_safe)
    result = np.zeros(n, dtype=bool)
    fallback = np.flatnonzero(~safe)
    for i in fallback:
        result[i] = robust_forbidden_intersection(A[i], B[i], S[i, :counts[i]])
    if return_diagnostics:
        return result, dict(pairs=n, interval_certified_safe=int(safe.sum()),
                            exact_fallback=len(fallback), forbidden=int(result.sum()))
    return result


def any_forbidden_intersection(V, faces_a, faces_b, pairs, shared_indices, shared_counts, *, chunk_size=8192):
    for start in range(0, len(pairs), chunk_size):
        p = pairs[start:start + chunk_size]
        if robust_forbidden_intersections(V[faces_a[p[:, 0]]], V[faces_b[p[:, 1]]],
                                          V[shared_indices[start:start + chunk_size]],
                                          shared_counts[start:start + chunk_size]).any():
            return True
    return False
