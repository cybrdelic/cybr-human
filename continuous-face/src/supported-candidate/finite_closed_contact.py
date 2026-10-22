"""Reconstructed finite closed-triangle oracle, not recovered original code.

Exact centroid/radius bounds retain every possible closest literal triangle.
Face/edge/angle-weighted-vertex pseudonormals sign ordinary nearest features;
ambiguous cases use all-face winding. Both assume a closed non-self-intersecting
owner; topology qualification alone does not prove that. Open surfaces are
never signed. A separate brute-force reference route supports bounded audits.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _finite_pairs(points, triangles):
    """Closest finite triangle points for an N x K candidate tensor."""
    p = np.asarray(points)[:, None, :]
    a, b, c = (triangles[:, :, k] for k in range(3))
    ab, ac = b-a, c-a
    scale = np.maximum(np.linalg.norm(ab, axis=-1), np.linalg.norm(ac, axis=-1))
    uedge, vedge, ap = ab/scale[..., None], ac/scale[..., None], (p-a)/scale[..., None]
    normal = np.cross(uedge, vedge)
    den = np.sum(normal*normal, axis=-1)
    u = np.sum(np.cross(ap, vedge)*normal, axis=-1)/den
    v = np.sum(np.cross(uedge, ap)*normal, axis=-1)/den
    bary = np.stack((1-u-v, u, v), axis=-1)
    q = a+u[..., None]*ab+v[..., None]*ac
    d2 = np.sum((q-p)**2, axis=-1)
    d2[np.min(bary, axis=-1) < 0] = np.inf
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, :, j]-triangles[:, :, i]
        fraction = np.clip(np.sum((p-triangles[:, :, i])*edge, axis=-1)/
                           np.sum(edge*edge, axis=-1), 0, 1)
        qq = triangles[:, :, i]+fraction[..., None]*edge
        dd = np.sum((qq-p)**2, axis=-1)
        choose = dd < d2
        bb = np.zeros_like(bary)
        bb[:, :, i], bb[:, :, j] = 1-fraction, fraction
        q[choose], d2[choose], bary[choose] = qq[choose], dd[choose], bb[choose]
    return q, d2, bary


class ClosedTriangleOwner:
    def __init__(self, vertices, faces, *, name="closed owner", owner_id=0):
        self.name, self.owner_id = str(name), int(owner_id)
        self.vertices = np.array(vertices, float, copy=True)
        raw = np.asarray(faces)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("integer M x 3 faces required")
        self.faces = np.array(raw, np.int64, copy=True)
        v, f = self.vertices, self.faces
        if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or
                len(f) < 4 or f.min() < 0 or f.max() >= len(v)):
            raise ValueError("finite referenced closed mesh required")
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        unique, inv, count = np.unique(np.sort(edges, axis=1), axis=0,
                                      return_inverse=True, return_counts=True)
        sense = np.bincount(inv, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
        graph = coo_matrix((np.ones(len(unique)), (unique[:, 0], unique[:, 1])),
                           shape=(len(v), len(v)))
        if (np.any(count != 2) or np.any(sense != 0) or
                connected_components(graph, directed=False, return_labels=False) != 1):
            raise ValueError("one closed consistently oriented manifold owner required")
        self.triangles = v[f]
        ab, ac = self.triangles[:, 1]-self.triangles[:, 0], self.triangles[:, 2]-self.triangles[:, 0]
        cross = np.cross(ab, ac)
        scale = np.maximum(np.linalg.norm(ab, axis=1), np.linalg.norm(ac, axis=1))
        rank = np.linalg.norm(cross, axis=1) / np.maximum(scale**2, np.finfo(float).tiny)
        if np.any(rank <= 1e-12):
            raise ValueError("unresolved/degenerate triangle; this prototype has no extended-precision fallback")
        centered = self.triangles - v.mean(axis=0)
        volume = np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum()/6
        if not np.isfinite(volume) or volume == 0:
            raise ValueError("nonzero finite oriented enclosed volume required")
        self.normals = np.sign(volume)*cross/np.linalg.norm(cross, axis=1)[:, None]
        self.face_edges = inv.reshape(3, -1).T
        self.edge_normals = np.zeros((len(unique), 3))
        self.vertex_normals = np.zeros_like(v)
        for k in range(3):
            np.add.at(self.edge_normals, self.face_edges[:, k], self.normals)
            aa = self.triangles[:, (k+1)%3]-self.triangles[:, k]
            bb = self.triangles[:, (k+2)%3]-self.triangles[:, k]
            angle = np.arctan2(np.linalg.norm(np.cross(aa, bb), axis=1), np.sum(aa*bb, axis=1))
            np.add.at(self.vertex_normals, f[:, k], angle[:, None]*self.normals)
        self.centers = self.triangles.mean(axis=1)
        self.radii = np.linalg.norm(self.triangles-self.centers[:, None], axis=2).max(axis=1)
        self.maximum_radius = float(self.radii.max())
        self.tree = cKDTree(self.centers)
        self.low, self.high = v.min(axis=0), v.max(axis=0)
        self.report = dict(name=self.name, owner_id=self.owner_id, triangles=len(f),
                           vertices=len(v), closed_oriented_manifold=True,
                           contact_mode="closed_signed",
                           minimum_scaled_triangle_rank=float(rank.min()),
                           source_self_intersections_checked=False,
                           reconstruction="Fresh finite-triangle/pseudonormal/winding implementation",
                           closest_coverage="Exact centroid/radius bound, not fixed-k nearest triangles")
        for a in vars(self).values():
            if isinstance(a, np.ndarray):
                a.flags.writeable = False
        self.tree.data.flags.writeable = False

    def query(self, points, chunk=128):
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("finite N x 3 query points required")
        values = {k: [] for k in ('points', 'signed_gap', 'gradient', 'valid', 'face_id', 'winding')}
        for start in range(0, len(p), chunk):
            x = p[start:start+chunk]
            k = min(8, len(self.faces))
            seed = np.asarray(self.tree.query(x, k=k)[1]).reshape(len(x), k)
            _, seed_d2, _ = _finite_pairs(x, self.triangles[seed])
            upper = np.sqrt(seed_d2.min(axis=1))
            pad = 128*np.finfo(float).eps*(np.linalg.norm(x, axis=1)+self.maximum_radius+upper+1)
            choices = self.tree.query_ball_point(x, upper+self.maximum_radius+pad)
            counts = np.array([len(c) for c in choices])
            pi, fi = np.repeat(np.arange(len(x)), counts), np.concatenate(choices).astype(int)
            keep = np.linalg.norm(self.centers[fi]-x[pi], axis=1)-self.radii[fi] <= upper[pi]+pad[pi]
            pi, fi = pi[keep], fi[keep]
            counts = np.bincount(pi, minlength=len(x))
            if np.any(counts == 0):
                raise FloatingPointError("conservative finite triangle coverage lost a seed")
            q, d2, bary = _finite_pairs(x[pi], self.triangles[fi, None])
            q, d2, bary = q[:, 0], d2[:, 0], bary[:, 0]
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            minimum = np.minimum.reduceat(d2, starts)
            winning_face = np.minimum.reduceat(np.where(d2 == minimum[pi], fi, len(self.faces)), starts)
            win = np.minimum.reduceat(np.where((d2 == minimum[pi]) & (fi == winning_face[pi]),
                                               np.arange(len(pi)), len(pi)), starts)
            closest, face, b = q[win], fi[win], bary[win]
            distance = np.sqrt(d2[win])
            active = b > 64*np.finfo(float).eps
            count = active.sum(axis=1)
            pseudo = self.normals[face].copy()
            for edge, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
                rows = (count == 2) & active[:, i] & active[:, j]
                pseudo[rows] = self.edge_normals[self.face_edges[face[rows], edge]]
            rows = count == 1
            pseudo[rows] = self.vertex_normals[self.faces[face[rows], np.argmax(b[rows], axis=1)]]
            length = np.linalg.norm(pseudo, axis=1)
            normal = pseudo/np.maximum(length[:, None], np.finfo(float).tiny)
            delta = x-closest
            oriented = np.sum(delta*normal, axis=1)
            resolution = 128*np.finfo(float).eps*(np.linalg.norm(x, axis=1)+
                                                 np.linalg.norm(closest, axis=1)+distance+1)
            surface = distance <= resolution
            sign = np.where(oriented < 0, -1., 1.)
            outside = np.any((x < self.low-resolution[:, None]) | (x > self.high+resolution[:, None]), axis=1)
            uncertain = ((length <= 64*np.finfo(float).eps) |
                         ((abs(oriented) <= resolution) & ~surface) | (outside & (sign < 0)))
            winding = np.full(len(x), np.nan)
            fallback = np.flatnonzero(uncertain & ~surface)
            if len(fallback):
                reference = self.query_bruteforce(x[fallback])
                winding[fallback] = reference['winding']
                sign[fallback] = np.where(reference['signed_gap'] < 0, -1., 1.)
                uncertain[fallback] = ~reference['valid']
            uncertain[surface] = False
            gradient = sign[:, None]*delta/np.maximum(distance[:, None], np.finfo(float).tiny)
            gradient[surface] = normal[surface]
            for key, data in zip(values, (closest, sign*distance, gradient, ~uncertain, face, winding)):
                values[key].append(data)
        if not len(p):
            return self.query_bruteforce(p)
        return {key: np.concatenate(value) for key, value in values.items()}

    def query_bruteforce(self, points, chunk=128):
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("finite N x 3 query points required")
        result = {k: [] for k in ("points", "signed_gap", "gradient", "valid", "face_id", "winding")}
        tri = self.triangles
        chunk = min(chunk, max(1, 65536//len(tri)))
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac = b-a, c-a
        scale = np.maximum(np.linalg.norm(ab, axis=1), np.linalg.norm(ac, axis=1))
        uedge, vedge = ab/scale[:, None], ac/scale[:, None]
        normal = np.cross(uedge, vedge)
        den = np.sum(normal*normal, axis=1)
        for start in range(0, len(p), chunk):
            x = p[start:start+chunk]
            ap = (x[:, None]-a)/scale[None, :, None]
            u = np.sum(np.cross(ap, vedge)*normal, axis=-1)/den
            v = np.sum(np.cross(uedge, ap)*normal, axis=-1)/den
            candidate = a + u[..., None]*ab + v[..., None]*ac
            d2 = np.sum((candidate-x[:, None])**2, axis=-1)
            d2[(u < 0) | (v < 0) | (u+v > 1)] = np.inf
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tri[:, j]-tri[:, i]
                fraction = np.clip(np.sum((x[:, None]-tri[:, i])*edge, axis=-1)/
                                   np.sum(edge*edge, axis=1), 0, 1)
                foot = tri[:, i] + fraction[..., None]*edge
                distance2 = np.sum((foot-x[:, None])**2, axis=-1)
                choose = distance2 < d2
                candidate[choose], d2[choose] = foot[choose], distance2[choose]
            face = np.argmin(d2, axis=1)
            closest = candidate[np.arange(len(x)), face]
            distance = np.sqrt(d2[np.arange(len(x)), face])
            # Generalized winding from all unchanged source triangles.
            aa, bb, cc = (tri[None]-x[:, None, None]).transpose(2, 0, 1, 3)
            la, lb, lc = (np.linalg.norm(y, axis=-1) for y in (aa, bb, cc))
            numerator = np.sum(aa*np.cross(bb, cc), axis=-1)
            denominator = (la*lb*lc + np.sum(aa*bb, axis=-1)*lc +
                           np.sum(bb*cc, axis=-1)*la + np.sum(cc*aa, axis=-1)*lb)
            winding = np.abs(np.sum(2*np.arctan2(numerator, denominator), axis=1)/(4*np.pi))
            valid = (winding < 1e-7) | (np.abs(winding-1) < 1e-7)
            sign = np.where(winding > .5, -1., 1.)
            pad = 128*np.finfo(float).eps*(np.max(np.abs(x), axis=1)+
                                          np.max(np.abs(self.vertices))+1)
            at_surface = distance <= pad
            valid[at_surface] = True  # Zero clearance is rejected by the barrier.
            gap = sign*distance
            gradient = sign[:, None]*(x-closest)/np.maximum(distance[:, None], np.finfo(float).tiny)
            gradient[at_surface] = self.normals[face[at_surface]]
            for k, value in zip(result, (closest, gap, gradient, valid, face, winding)):
                result[k].append(value)
        if not len(p):
            return dict(points=np.empty((0, 3)), signed_gap=np.empty(0), gradient=np.empty((0, 3)),
                        valid=np.empty(0, bool), face_id=np.empty(0, int), winding=np.empty(0))
        return {k: np.concatenate(value) for k, value in result.items()}


class UnsignedTriangleSheetOwner:
    """Explicit two-sided finite sheet, with NO inferred inside/outside sign.

    Any topology is permitted if every literal triangle is numerically resolved.
    Distance to this fixed triangle set is 1-Lipschitz. Positive sampled path
    distance prevents crossing a represented triangle along an accepted straight
    segment. It does not establish the correct starting side, close holes, or
    prohibit paths around sheet boundaries. No triangles are removed or added.
    """
    def __init__(self, vertices, faces, *, name="unsigned finite sheet", owner_id=0):
        self.name, self.owner_id = str(name), int(owner_id)
        self.vertices = np.array(vertices, float, copy=True)
        raw = np.asarray(faces)
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("integer M x 3 sheet faces required")
        self.faces = np.array(raw, np.int64, copy=True)
        v, f = self.vertices, self.faces
        if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or
                not len(f) or f.min() < 0 or f.max() >= len(v)):
            raise ValueError("finite referenced triangle sheet required")
        self.triangles = v[f]
        ab, ac = self.triangles[:, 1]-self.triangles[:, 0], self.triangles[:, 2]-self.triangles[:, 0]
        scale = np.maximum(np.linalg.norm(ab, axis=1), np.linalg.norm(ac, axis=1))
        rank = np.linalg.norm(np.cross(ab, ac), axis=1)/np.maximum(scale**2, np.finfo(float).tiny)
        if np.any(rank <= 1e-12):
            raise ValueError("unresolved/degenerate sheet triangle; no triangles silently removed")
        self.centers = self.triangles.mean(axis=1)
        self.radii = np.linalg.norm(self.triangles-self.centers[:, None], axis=2).max(axis=1)
        self.maximum_radius = float(self.radii.max())
        self.tree = cKDTree(self.centers)
        self.low, self.high = v.min(axis=0), v.max(axis=0)
        edge = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        _, count = np.unique(np.sort(edge, axis=1), axis=0, return_counts=True)
        self.report = dict(name=self.name, owner_id=self.owner_id, contact_mode="unsigned_sheet",
                           triangles=len(f), vertices=len(v), boundary_edges=int((count == 1).sum()),
                           nonmanifold_edges=int((count > 2).sum()),
                           minimum_scaled_triangle_rank=float(rank.min()), inside_sign_supplied=False,
                           starting_side_validated=False, triangles_removed=0,
                           limits="Finite two-sided sheet only; no enclosure, hole closure or initial-side certificate")
        for a in vars(self).values():
            if isinstance(a, np.ndarray):
                a.flags.writeable = False
        self.tree.data.flags.writeable = False

    def query(self, points, chunk=128):
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("finite N x 3 sheet query points required")
        out = {k: [] for k in ('points', 'signed_gap', 'gradient', 'valid', 'face_id', 'winding')}
        for start in range(0, len(p), chunk):
            x = p[start:start+chunk]
            k = min(8, len(self.faces))
            seed = np.asarray(self.tree.query(x, k=k)[1]).reshape(len(x), k)
            _, seed_d2, _ = _finite_pairs(x, self.triangles[seed])
            upper = np.sqrt(seed_d2.min(axis=1))
            pad = 128*np.finfo(float).eps*(np.linalg.norm(x, axis=1)+self.maximum_radius+upper+1)
            choices = self.tree.query_ball_point(x, upper+self.maximum_radius+pad)
            counts = np.array([len(c) for c in choices])
            pi, fi = np.repeat(np.arange(len(x)), counts), np.concatenate(choices).astype(int)
            keep = np.linalg.norm(self.centers[fi]-x[pi], axis=1)-self.radii[fi] <= upper[pi]+pad[pi]
            pi, fi = pi[keep], fi[keep]
            counts = np.bincount(pi, minlength=len(x))
            if np.any(counts == 0):
                raise FloatingPointError("finite sheet coverage lost a seed")
            q, d2, _ = _finite_pairs(x[pi], self.triangles[fi, None])
            q, d2 = q[:, 0], d2[:, 0]
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            minimum = np.minimum.reduceat(d2, starts)
            winning_face = np.minimum.reduceat(np.where(d2 == minimum[pi], fi, len(self.faces)), starts)
            win = np.minimum.reduceat(np.where((d2 == minimum[pi]) & (fi == winning_face[pi]),
                                               np.arange(len(pi)), len(pi)), starts)
            closest, face, distance = q[win], fi[win], np.sqrt(d2[win])
            gradient = (x-closest)/np.maximum(distance[:, None], np.finfo(float).tiny)
            # signed_gap is a common interface key only; here it is UNSIGNED.
            for key, data in zip(out, (closest, distance, gradient, np.ones(len(x), bool),
                                       face, np.full(len(x), np.nan))):
                out[key].append(data)
        if not len(p):
            return dict(points=np.empty((0, 3)), signed_gap=np.empty(0), gradient=np.empty((0, 3)),
                        valid=np.empty(0, bool), face_id=np.empty(0, int), winding=np.empty(0))
        return {key: np.concatenate(value) for key, value in out.items()}
