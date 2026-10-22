"""Exact endpoint intersection for triangles with explicitly owned shared features.

``robust_forbidden_intersection(A, B, shared_points)`` returns whether the two
closed triangles intersect outside their supplied shared vertex or shared edge.
The ownership must be supplied, not inferred from proximity. Every supplied
point must be an exactly equal vertex of both triangles. Degenerate triangles
fail closed (return True); malformed/non-finite inputs or invalid ownership
raise ValueError. This is an endpoint predicate, not a continuous-path test.

All finite binary64 coordinates are lifted onto ONE common dyadic integer
scale *before subtraction*. All subsequent arithmetic uses Python integers.
There is no angular, distance, area, or ownership tolerance and no reconstructed
floating-point intersection. Even subnormal or overflow-sized coordinates work.

Proof of the shared-vertex reduction
-----------------------------------
Translate the owned common vertex to the origin. A nondegenerate triangle with
remaining vertices u,v is {a*u+b*v: a,b >= 0, a+b <= 1}. Two such triangles
intersect away from zero iff their cones {a*u+b*v: a,b >= 0} have a common
nonzero ray: scale any common conic vector down sufficiently to meet both sum
constraints. Thus no remote intersection can be missed by testing the cones.

For nonparallel plane normals nA,nB the only possible common directions are
+d and -d, d = nA cross nB. A direction w already in span(u,v) lies in its cone
iff (u cross w).n >= 0 and (w cross v).n >= 0, n = u cross v. These quantities
are the two conic coefficients times n.n. When the planes coincide, the two
pointed planar wedges overlap iff a boundary ray of either lies in the other.
All boundaries are closed, so collinear segment overlap is forbidden too.

For a shared edge, distinct planes intersect only on the edge's supporting
line, and each triangle meets that line exactly along the edge. In one plane,
the triangles intersect beyond the edge iff the two third vertices lie on its
same side. These are exact determinant/sign decisions.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

Point = tuple[int, int, int]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Point, b: Point) -> Point:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a: Point, b: Point) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _points(value: Iterable[Sequence[float]], name: str,
            allowed_counts: tuple[int, ...]) -> list[tuple[float, float, float]]:
    try:
        rows = [tuple(float(x) for x in row) for row in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain finite binary64 3D points') from exc
    if len(rows) not in allowed_counts or any(len(row) != 3 for row in rows):
        raise ValueError(f'{name} must have shape ({" or ".join(map(str, allowed_counts))}, 3)')
    if not all(math.isfinite(x) for row in rows for x in row):
        raise ValueError(f'{name} must contain only finite coordinates')
    return rows


def _lift(points: list[tuple[float, float, float]]) -> list[Point]:
    """Map binary64 values exactly to one common positive dyadic scale."""
    ratios = [[x.as_integer_ratio() for x in row] for row in points]
    scale = max(denominator for row in ratios for _, denominator in row)
    # Each denominator is a power of two, so every division is exact.
    return [tuple(numerator * (scale // denominator)
                  for numerator, denominator in row) for row in ratios]


def _in_cone(w: Point, u: Point, v: Point, normal: Point) -> bool:
    """Closed cone membership; caller guarantees w is in its plane."""
    return _dot(_cross(u, w), normal) >= 0 and _dot(_cross(w, v), normal) >= 0


def robust_forbidden_intersection(A, B, shared_points) -> bool:
    """Return True for intersection beyond an explicitly owned vertex/edge.

    Coordinates are interpreted as their binary64 values. Supply shared_points
    as shape (1,3) or (2,3). Ownership mismatch, duplicate owned points, zero or
    three owned points, wrong shapes, and non-finite data raise ValueError.
    Exact zero-area triangles return True, regardless of their relative pose.
    """
    aa = _points(A, 'A', (3,))
    bb = _points(B, 'B', (3,))
    ss = _points(shared_points, 'shared_points', (1, 2))
    if len(set(ss)) != len(ss):
        raise ValueError('Owned shared points must be distinct')
    if any(s not in aa or s not in bb for s in ss):
        raise ValueError('Every owned shared point must exactly equal a vertex of both triangles')

    points = _lift(aa + bb + ss)
    a, b, shared = points[:3], points[3:6], points[6:]
    na = _cross(_sub(a[1], a[0]), _sub(a[2], a[0]))
    nb = _cross(_sub(b[1], b[0]), _sub(b[2], b[0]))
    if not any(na) or not any(nb):
        return True

    origin = shared[0]
    if len(shared) == 2:
        edge = _sub(shared[1], origin)
        au = _sub(next(p for p in a if p not in shared), origin)
        bu = _sub(next(p for p in b if p not in shared), origin)
        side_a, side_b = _cross(edge, au), _cross(edge, bu)
        if _dot(side_a, bu) != 0:
            return False  # Noncoplanar: only the explicitly owned edge meets.
        return _dot(side_a, side_b) > 0

    au, av = [_sub(p, origin) for p in a if p != origin]
    bu, bv = [_sub(p, origin) for p in b if p != origin]
    na, nb = _cross(au, av), _cross(bu, bv)
    direction = _cross(na, nb)
    if any(direction):
        if _in_cone(direction, au, av, na) and _in_cone(direction, bu, bv, nb):
            return True
        opposite = (-direction[0], -direction[1], -direction[2])
        return _in_cone(opposite, au, av, na) and _in_cone(opposite, bu, bv, nb)

    return (_in_cone(au, bu, bv, nb) or _in_cone(av, bu, bv, nb)
            or _in_cone(bu, au, av, na) or _in_cone(bv, au, av, na))
