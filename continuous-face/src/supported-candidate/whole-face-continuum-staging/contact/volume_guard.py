"""Outward-rounded Bernstein determinant proof for every affine carrier path.

No time samples certify a path. det(F(t)) is a cubic; interval Bernstein
coefficients bound it on the complete subinterval. Ambiguity fails closed.
"""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deformable-contact-staging'))
from moving_triangle_contact import _down, _up, _iadd, _imul


def neg(a):
    return -a[1], -a[0]


def determinant(a, b, c):
    out = (np.zeros(a[0].shape[:-1]), np.zeros(a[0].shape[:-1]))
    for i, j, k, sign in ((0,1,2,1),(1,2,0,1),(2,0,1,1),
                           (0,2,1,-1),(1,0,2,-1),(2,1,0,-1)):
        term = _imul(_imul((a[0][...,i],a[1][...,i]),
                            (b[0][...,j],b[1][...,j])),
                       (c[0][...,k],c[1][...,k]))
        out = _iadd(out, term if sign > 0 else neg(term))
    return out


def deformation_interval(solid, q):
    q = np.asarray(q, float)
    if q.shape != solid.X.shape or not np.isfinite(q).all():
        raise ValueError('finite shape-matched coordinates required')
    d = np.stack([q[solid.T[:,k]]-q[solid.T[:,0]] for k in (1,2,3)], axis=2)
    dlo, dhi = _down(d), _up(d)
    out = (np.zeros_like(d), np.zeros_like(d))
    for k in range(3):
        term = _imul((dlo[:,:,k,None], dhi[:,:,k,None]),
                     (solid.inv[:,None,k,:], solid.inv[:,None,k,:]))
        out = _iadd(out, term)
    return out


def bernstein_coefficients(solid, q0, q1):
    F = deformation_interval(solid, q0)
    G = deformation_interval(solid, q1)
    f = [(F[0][...,j], F[1][...,j]) for j in range(3)]
    g = [(G[0][...,j], G[1][...,j]) for j in range(3)]
    b0, b3 = determinant(*f), determinant(*g)
    b1 = _iadd(_iadd(determinant(g[0], f[1], f[2]),
                            determinant(f[0], g[1], f[2])),
                            determinant(f[0], f[1], g[2]))
    b2 = _iadd(_iadd(determinant(f[0], g[1], g[2]),
                            determinant(g[0], f[1], g[2])),
                            determinant(g[0], g[1], f[2]))
    b1 = (_down(b1[0]/3), _up(b1[1]/3))
    b2 = (_down(b2[0]/3), _up(b2[1]/3))
    return (np.stack([b[0] for b in (b0,b1,b2,b3)],axis=1),
            np.stack([b[1] for b in (b0,b1,b2,b3)],axis=1))


def _mid(a, b):
    c = _iadd(a,b)
    return _down(c[0]*.5), _up(c[1]*.5)


def split(B):
    b = [(B[0][:,j],B[1][:,j]) for j in range(4)]
    a = [_mid(b[j],b[j+1]) for j in range(3)]
    c = [_mid(a[j],a[j+1]) for j in range(2)]
    d = _mid(c[0],c[1])
    pack = lambda v: (np.stack([x[0] for x in v],axis=1),
                      np.stack([x[1] for x in v],axis=1))
    return pack((b[0],a[0],c[0],d)), pack((d,c[1],a[2],b[3]))


def certify_positive_volume(solid, q0, q1, *, floor=.1, max_depth=20,
                            max_interval_rows=250000):
    if not 0 <= floor < 1:
        raise ValueError('Jacobian floor must be in [0,1)')
    try:
        B = bernstein_coefficients(solid, q0, q1)
    except (ValueError,FloatingPointError) as exc:
        return dict(certified=False, reason=str(exc), jacobian_floor=floor)
    if not np.isfinite(B).all():
        return dict(certified=False, reason='nonfinite determinant enclosure', jacobian_floor=floor)
    stack = [(B,np.arange(len(solid.T)),0)]
    rows, visits, maximum_depth, certified_lower = 0,0,0,np.inf
    while stack:
        B,ids,depth = stack.pop(); visits += 1; rows += len(ids)
        maximum_depth = max(maximum_depth,depth)
        lo,hi = B
        good = lo.min(1) > floor
        if np.any(good):
            certified_lower = min(certified_lower,float(lo[good].min()))
        bad = ~good
        if not np.any(bad):
            continue
        fail = (hi[:,0] <= floor)|(hi[:,-1] <= floor)|(hi.max(1) <= floor)
        if np.any(fail) or depth >= max_depth or rows > max_interval_rows:
            selected = np.flatnonzero(fail if np.any(fail) else bad)[0]
            return dict(certified=False,reason='nonpositive or unresolved whole-path Jacobian',
                        failing_carrier_tet=int(ids[selected]), jacobian_floor=floor,
                        subdivision_depth=maximum_depth, interval_rows=rows)
        left,right = split((lo[bad],hi[bad]))
        stack.append((right,ids[bad],depth+1)); stack.append((left,ids[bad],depth+1))
    return dict(certified=True, jacobian_floor=floor,
                conservative_minimum_J=float(certified_lower),
                subdivision_depth=maximum_depth, interval_rows=rows,
                subinterval_visits=visits, carrier_tets=len(solid.T),
                guarantee='outward-rounded cubic Bernstein determinant bounds over the entire affine path')
