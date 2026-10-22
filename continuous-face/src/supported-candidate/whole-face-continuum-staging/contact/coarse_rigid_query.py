"""Native rigid finite-site queries for the explicitly coarse physical model.

This is NOT a boundary collision certificate. Fixed-k finite-triangle distance
is an upper-bound approximation, not an exact global closest distance. Closed,
consistently oriented native owners get independent finite z-ray occupancy;
open/nonmanifold owners stay unsigned two-sided sheets. No source facets move,
are removed, or are filled. Input and output length units match coordinate_scale.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
HELPERS = HERE.parent/'geometry'/'helpers'
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))
from mesh_queries import SurfaceIndex, closest_surface
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from finite_closed_contact import _finite_pairs


def _bins(triangles, dimensions=3):
    t = np.asarray(triangles)[..., :dimensions]
    centers = t.mean(1)
    radii = np.linalg.norm(t-centers[:, None], axis=2).max(1)
    levels = np.floor(np.log2(np.maximum(radii, 1e-12))).astype(int)
    return [(np.flatnonzero(levels == level),
             cKDTree(centers[levels == level]),
             float(radii[levels == level].max())) for level in np.unique(levels)]


class NativeRigidQuery:
    """query(points) returns arrays signed_gap, normal, owner_id, finite_valid.

    Additional aliases gradient and valid support existing contact callers.
    Query results have point-site scope only. Rays with unresolved projected
    edge hits fail finite_valid rather than silently guessing an inside sign.
    """
    def __init__(self, vertices, faces, owner_ids, *, records=None,
                 candidate_count=48, source_report=None):
        started = time.perf_counter()
        self.vertices = np.asarray(vertices, float).copy()
        self.faces = np.asarray(faces, np.int64).copy()
        self.face_owner = np.asarray(owner_ids, np.int64).copy()
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or
                not np.isfinite(self.vertices).all() or self.faces.ndim != 2 or
                self.faces.shape[1] != 3 or not len(self.faces) or
                self.face_owner.shape != (len(self.faces),) or self.faces.min() < 0 or
                self.faces.max() >= len(self.vertices)):
            raise ValueError('Finite native vertices, referenced triangles and per-face owners required')
        self.candidate_count = int(candidate_count)
        if self.candidate_count < 1:
            raise ValueError('Positive candidate_count required')
        self.index = SurfaceIndex(self.vertices, self.faces)
        self.triangles = self.index.triangles
        self.centers = self.triangles.mean(1)
        self.radii = np.linalg.norm(self.triangles-self.centers[:, None], axis=2).max(1)
        self.distance_bins = _bins(self.triangles)
        self.owners = {}
        records = {int(r['part_id']): r for r in (records or [])}
        self.owner_ids = np.unique(self.face_owner)
        self.owner_slot = np.searchsorted(self.owner_ids, self.face_owner)
        cross = np.cross(self.triangles[:, 1]-self.triangles[:, 0], self.triangles[:, 2]-self.triangles[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        self.face_normals = cross/np.maximum(area2[:, None], np.finfo(float).tiny)
        self.resolved_faces = area2 > 0
        owner_reports = []
        for pid in self.owner_ids:
            ids = np.flatnonzero(self.face_owner == pid)
            f = self.faces[ids]
            edge = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
            _, inv, count = np.unique(np.sort(edge, axis=1), axis=0, return_counts=True, return_inverse=True)
            sense = np.bincount(inv, weights=np.where(edge[:, 0] < edge[:, 1], 1, -1))
            tri = self.triangles[ids]
            ab, ac = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
            scale = np.maximum(np.linalg.norm(ab, axis=1), np.linalg.norm(ac, axis=1))
            rank = area2[ids]/np.maximum(scale**2, np.finfo(float).tiny)
            resolved = bool(np.all(rank > 1e-12))
            closed = bool(np.all(count == 2) and np.all(sense == 0) and resolved)
            self.owners[int(pid)] = dict(ids=ids, closed=closed, low=tri.min((0, 1)),
                                        high=tri.max((0, 1)), index=None)
            r = records.get(int(pid), {})
            owner_reports.append(dict(owner_id=int(pid), name=r.get('name', str(pid)),
                kind=r.get('kind', 'fixture'), triangles=len(ids),
                contact_mode='closed_signed_finite_ray' if closed else 'unsigned_two_sided_finite_sheet',
                boundary_edges=int(np.sum(count == 1)), nonmanifold_edges=int(np.sum(count > 2)),
                inconsistent_oriented_edges=int(np.sum(sense != 0)),
                unresolved_or_degenerate_triangles=int(np.sum(rank <= 1e-12)),
                source_self_intersections_checked=False))
        self.closed_faces = np.flatnonzero(np.isin(self.face_owner, [p for p, o in self.owners.items() if o['closed']]))
        self.ray_triangles = self.triangles[self.closed_faces]
        self.ray_bins = _bins(self.ray_triangles, 2) if len(self.closed_faces) else []
        self._oblique = None
        self.report = dict(source_report or {}, owners=owner_reports, owner_count=len(self.owners),
            triangles=len(self.faces), candidate_count=self.candidate_count,
            distance_method='Finite closest points on fixed-k triangle-centroid candidates, upper-bound approximation only',
            sign_method='Independent per-closed-owner finite native z-ray nonzero winding; conservative projected-radius candidates; oblique retry for edge ambiguities',
            inactive_bound='Conservative minimum over 3D radius bins of nearest centroid distance minus bin maximum radius',
            unresolved_facets='Preserved unchanged; finite segment fallback in mesh_queries.closest_surface; owners with unresolved facets not signed',
            open_owner_occupancy='Not supplied: two-sided finite sheets are not invented solids',
            surface_collision_certified=False, source_self_intersections_checked=False,
            site_resolution='Caller-defined positive-volume integration sites; no whole-cell or between-site guarantee',
            source_triangles_removed=0, source_triangles_added=0,
            build_seconds=time.perf_counter()-started)

    @classmethod
    def from_native(cls, candidate_count=48, coordinate_scale=1.0, source_path=None,
                    metadata_path=None):
        source = Path(source_path or BASE/'full-face-coverage-plan/restored/exact-rest/anatomy-registered-rest.npz')
        metadata = Path(metadata_path or BASE/'full-face-coverage-plan/restored/anatomy-reference/prior-body-handoff/anatomy-metadata.json')
        scale = float(coordinate_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Positive coordinate_scale required')
        packet = np.load(source, allow_pickle=False)
        meta = json.loads(metadata.read_text())
        records = [dict(p, part_id=i) for i, p in enumerate(meta['parts'])
                   if ((p['kind'] in ('bone', 'cartilage') and i < 193) or p['kind'] == 'teeth')]
        take = np.isin(packet['part_ids'], [r['part_id'] for r in records])
        return cls(packet['vertices']*scale, packet['triangles'][take], packet['part_ids'][take],
            records=records, candidate_count=candidate_count,
            source_report=dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(), coordinate_scale=scale,
                inventory_rule='(kind in bone/cartilage and part_id < 193) or kind == teeth',
                geometry_modified=False))

    def distance_lower_bound(self, points):
        """Conservative unsigned distance lower bound, NOT a signed clearance.

        For active-band selection require bound > activation + motion budget.
        This does not establish initial exterior membership on its own.
        """
        p = self._points(points)
        result = np.full(len(p), np.inf)
        pad = 256*np.finfo(float).eps*(np.linalg.norm(p, axis=1)+np.max(abs(self.vertices))+1)
        for _, tree, radius in self.distance_bins:
            result = np.minimum(result, tree.query(p, k=1)[0]-radius-pad)
        return np.maximum(result, 0.)

    def clipped_distance_lower_bound(self, points, cutoff, chunk=256):
        """Tight finite-site distance bound clipped at cutoff, in input units.

        All source triangles that could be within cutoff are included by
        conservative centroid/radius search then triangle-AABB filtering.
        Numerical finite-triangle distances reuse finite_closed_contact._finite_pairs,
        with a finite edge fallback for exactly degenerate triangles.
        This is a floating-point point-site bound, not a global collision proof.
        """
        p = self._points(points)
        cutoff = float(cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('Positive finite cutoff required')
        result = np.full(len(p), cutoff)
        for start in range(0, len(p), chunk):
            x = p[start:start+chunk]
            pad = 1024*np.finfo(float).eps*(np.linalg.norm(x, axis=1)+np.max(abs(self.vertices))+cutoff+1)
            minimum = np.full(len(x), cutoff**2)
            for ids, tree, radius in self.distance_bins:
                choices = tree.query_ball_point(x, radius+cutoff+pad)
                count = np.fromiter(map(len, choices), int, len(x))
                if not count.sum():
                    continue
                pi = np.repeat(np.arange(len(x)), count)
                fi = ids[np.concatenate(choices).astype(int)]
                for block in range(0, len(pi), 16384):
                    ii, jj = pi[block:block+16384], fi[block:block+16384]
                    tt = self.triangles[jj]
                    box_delta = np.maximum(np.maximum(tt.min(1)-x[ii], x[ii]-tt.max(1)), 0.)
                    keep = np.sum(box_delta**2, axis=1) <= (cutoff+pad[ii])**2
                    ii, jj, tt = ii[keep], jj[keep], tt[keep]
                    if not len(ii):
                        continue
                    good = self.resolved_faces[jj]
                    d2 = np.full(len(ii), np.inf)
                    if np.any(good):
                        _, dd, _ = _finite_pairs(x[ii[good]], tt[good, None])
                        d2[good] = dd[:, 0]
                    # Exactly degenerate native facets remain their finite edges.
                    if np.any(~good):
                        pp, triangles = x[ii[~good]], tt[~good]
                        edge_d2 = np.full(len(pp), np.inf)
                        for ia, ib in ((0, 1), (1, 2), (2, 0)):
                            edge = triangles[:, ib]-triangles[:, ia]
                            den = np.sum(edge*edge, axis=1)
                            fraction = np.clip(np.divide(np.sum((pp-triangles[:, ia])*edge, axis=1), den,
                                out=np.zeros(len(pp)), where=den>0), 0, 1)
                            foot = triangles[:, ia]+fraction[:, None]*edge
                            edge_d2 = np.minimum(edge_d2, np.sum((foot-pp)**2, axis=1))
                        d2[~good] = edge_d2
                    np.minimum.at(minimum, ii, d2)
            result[start:start+len(x)] = np.maximum(np.sqrt(minimum)-pad, 0.)
        return result

    @staticmethod
    def _points(points):
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError('Finite M x 3 query points required')
        return p

    def _ray(self, p, tri, bins, bounds=None):
        """Nonzero crossing count per owner with edge-hit ambiguity detection."""
        count = np.zeros((len(p), len(self.owner_ids)), np.int16)
        ambiguous = np.zeros(len(p), bool)
        if not len(tri) or not len(p):
            return count != 0, ~ambiguous
        pad = 256*np.finfo(float).eps*(np.max(abs(p), axis=1)+np.max(abs(tri))+1)
        for ids, tree, radius in bins:
            choices = tree.query_ball_point(p[:, :2], radius+pad)
            n = np.fromiter(map(len, choices), int, len(p))
            if not n.sum():
                continue
            pi = np.repeat(np.arange(len(p)), n)
            fi = ids[np.concatenate(choices).astype(int)]
            t = tri[fi]
            keep = np.all((p[pi, :2] >= t[:, :, :2].min(1)-pad[pi, None]) &
                          (p[pi, :2] <= t[:, :, :2].max(1)+pad[pi, None]), axis=1)
            pi, fi, t = pi[keep], fi[keep], t[keep]
            if not len(pi):
                continue
            ab, ac, delta = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0], p[pi]-t[:, 0]
            det = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
            projected = abs(det) > 64*np.finfo(float).eps*np.maximum(np.linalg.norm(ab[:, :2], axis=1)*np.linalg.norm(ac[:, :2], axis=1), 1e-30)
            u = np.divide(delta[:, 0]*ac[:, 1]-delta[:, 1]*ac[:, 0], det,
                          out=np.full(len(pi), -10.), where=projected)
            v = np.divide(ab[:, 0]*delta[:, 1]-ab[:, 1]*delta[:, 0], det,
                          out=np.full(len(pi), -10.), where=projected)
            tol = 1e-10
            hit = projected & (u >= -tol) & (v >= -tol) & (u+v <= 1+tol)
            z = t[:, 0, 2]+u*ab[:, 2]+v*ac[:, 2]
            below = z <= p[pi, 2]+pad[pi]
            edge = hit & below & ((abs(u) <= tol) | (abs(v) <= tol) | (abs(1-u-v) <= tol) | (abs(z-p[pi, 2]) <= pad[pi]))
            ambiguous[pi[edge]] = True
            hit &= below
            slots = self.owner_slot[self.closed_faces[fi[hit]]]
            np.add.at(count, (pi[hit], slots), np.sign(det[hit]).astype(np.int16))
        # Finite owner bounds prevent extending a malformed numerical ray outside native bounds.
        for slot, pid in enumerate(self.owner_ids):
            owner = self.owners[int(pid)]
            low, high = (owner['low'], owner['high']) if bounds is None else bounds[slot]
            outside = np.any((p < low-pad[:, None]) | (p > high+pad[:, None]), axis=1)
            count[outside, slot] = 0
        return count != 0, ~ambiguous

    def containment(self, points, chunk=2048):
        """Return (inside[M, owner_count], valid[M]); open owners always false."""
        p = self._points(points)
        all_inside, all_valid = [], []
        for start in range(0, len(p), chunk):
            x = p[start:start+chunk]
            inside, valid = self._ray(x, self.ray_triangles, self.ray_bins)
            # Retry only ambiguous rays in a generic orthonormal frame. No geometry movement.
            bad = np.flatnonzero(~valid)
            if len(bad):
                if self._oblique is None:
                    direction = np.array([.317, .119, 1.]); direction /= np.linalg.norm(direction)
                    e1 = np.cross(direction, [0., 1., 0.]); e1 /= np.linalg.norm(e1)
                    e2 = np.cross(direction, e1)
                    matrix = np.column_stack((e1, e2, direction))
                    transformed = self.ray_triangles@matrix
                    bounds = []
                    for pid in self.owner_ids:
                        tt = self.triangles[self.owners[int(pid)]['ids']]@matrix
                        bounds.append((tt.min((0, 1)), tt.max((0, 1))))
                    self._oblique = matrix, transformed, _bins(transformed, 2), bounds
                matrix, transformed, bins, bounds = self._oblique
                ii, vv = self._ray(x[bad]@matrix, transformed, bins, bounds=bounds)
                inside[bad], valid[bad] = ii, vv
            all_inside.append(inside); all_valid.append(valid)
        if not len(p):
            return np.zeros((0, len(self.owner_ids)), bool), np.zeros(0, bool)
        return np.concatenate(all_inside), np.concatenate(all_valid)

    def _finite_witness(self, points, face_ids):
        """Paired finite source-triangle witnesses, preserving degenerate edges."""
        tt = self.triangles[face_ids]
        good = self.resolved_faces[face_ids]
        q = np.zeros_like(points)
        d2 = np.full(len(points), np.inf)
        if np.any(good):
            qq, dd, _ = _finite_pairs(points[good], tt[good, None])
            q[good], d2[good] = qq[:, 0], dd[:, 0]
        if np.any(~good):
            p, t = points[~good], tt[~good]
            best = np.full(len(p), np.inf)
            witness = np.zeros_like(p)
            for ia, ib in ((0, 1), (1, 2), (2, 0)):
                edge = t[:, ib]-t[:, ia]
                den = np.sum(edge*edge, axis=1)
                fraction = np.clip(np.divide(np.sum((p-t[:, ia])*edge, axis=1), den,
                    out=np.zeros(len(p)), where=den>0), 0, 1)
                foot = t[:, ia]+fraction[:, None]*edge
                dd = np.sum((foot-p)**2, axis=1)
                take = dd < best
                best[take], witness[take] = dd[take], foot[take]
            q[~good], d2[~good] = witness, best
        return q, d2

    def query_band(self, points, clearance=.1, penetration_limit=.05, chunk=256, owner_terms=False):
        """Finite exact-within-band point energy/gradient queries, not fixed-k.

        Exterior sites farther than clearance return gap=clearance, normal=0,
        owner=-1 and face=-1. Independent native closed-owner containment still
        runs. Every containing owner's closest finite distance is considered;
        deepest penetration wins. A missing containing-owner witness proves
        depth exceeds the band, and fails validity rather than being hidden by
        another owner's nearby surface. Over-limit penetration also fails.
        All claims are ordinary floating-point finite-site claims, not exact
        predicates, whole-cell occupancy, CCD or boundary collision certificates.
        """
        p = self._points(points)
        clearance, penetration_limit = float(clearance), float(penetration_limit)
        if not np.isfinite(clearance) or clearance <= 0 or not np.isfinite(penetration_limit) or penetration_limit < 0:
            raise ValueError('Positive clearance and nonnegative penetration limit required')
        cutoff = max(clearance, penetration_limit)
        owner_count = len(self.owner_ids)
        rows = []
        self.report['band_query'] = dict(distance_method='All finite triangles within max(clearance, penetration_limit), conservative radius/AABB candidates',
            force_method='Finite closest-feature witness derivative; no fixed-k candidate jumps inside activation band',
            beyond_band='Exterior zero energy; containing owner without witness fails closed',
            floating_point_only=True)
        for start in range(0, len(p), chunk):
            x = p[start:start+chunk]
            pad = 1024*np.finfo(float).eps*(np.linalg.norm(x, axis=1)+np.max(abs(self.vertices))+cutoff+1)
            minimum = np.full(len(x)*owner_count, np.inf)
            witness = np.zeros((len(minimum), 3))
            winface = np.full(len(minimum), len(self.faces), int)
            for ids, tree, radius in self.distance_bins:
                choices = tree.query_ball_point(x, radius+cutoff+pad)
                count = np.fromiter(map(len, choices), int, len(x))
                if not count.sum():
                    continue
                pi = np.repeat(np.arange(len(x)), count)
                fi = ids[np.concatenate(choices).astype(int)]
                for block in range(0, len(pi), 16384):
                    ii, jj = pi[block:block+16384], fi[block:block+16384]
                    tt = self.triangles[jj]
                    delta = np.maximum(np.maximum(tt.min(1)-x[ii], x[ii]-tt.max(1)), 0.)
                    keep = np.sum(delta**2, axis=1) <= (cutoff+pad[ii])**2
                    ii, jj = ii[keep], jj[keep]
                    if not len(ii):
                        continue
                    qq, dd = self._finite_witness(x[ii], jj)
                    keep = dd <= (cutoff+pad[ii])**2
                    ii, jj, qq, dd = ii[keep], jj[keep], qq[keep], dd[keep]
                    if not len(ii):
                        continue
                    key = ii*owner_count+self.owner_slot[jj]
                    order = np.lexsort((jj, dd, key))
                    order = order[np.r_[True, key[order][1:] != key[order][:-1]]]
                    keys = key[order]
                    better = (dd[order] < minimum[keys]) | ((dd[order] == minimum[keys]) & (jj[order] < winface[keys]))
                    selected, keys = order[better], keys[better]
                    minimum[keys], witness[keys], winface[keys] = dd[selected], qq[selected], jj[selected]
            distances = np.sqrt(minimum.reshape(len(x), owner_count))
            inside, ray_valid = self.containment(x, chunk=chunk)
            rr = np.arange(len(x))
            chosen = np.argmin(distances, axis=1)
            any_inside = inside.any(1)
            if np.any(any_inside):
                chosen[any_inside] = np.argmax(np.where(inside[any_inside], distances[any_inside], -np.inf), axis=1)
            distance = distances[rr, chosen]
            missing_inside = any_inside & ~np.isfinite(distance)
            # A finite negative sentinel is explicitly invalid, never used as a physical force.
            gap = np.where(any_inside, -distance, np.minimum(distance, clearance))
            gap[missing_inside] = -(cutoff+np.maximum(pad[missing_inside], 1e-12))
            active = np.isfinite(distance) & (any_inside | (distance < clearance))
            key = rr*owner_count+chosen
            q = x.copy(); q[active] = witness[key[active]]
            face = np.full(len(x), -1, int); face[active] = winface[key[active]]
            owner = np.full(len(x), -1, int); owner[active | any_inside] = self.owner_ids[chosen[active | any_inside]]
            normal = np.zeros_like(x)
            nonzero = active & (distance > 1e-13)
            normal[nonzero] = (x[nonzero]-q[nonzero])/distance[nonzero, None]
            normal[nonzero & any_inside] *= -1
            on = active & ~nonzero
            normal[on] = self.face_normals[face[on]]
            exceeded = any_inside & ((distance > penetration_limit+pad) | missing_inside)
            finite = ray_valid & ~exceeded & np.isfinite(gap) & np.isfinite(normal).all(1)
            rows.append(dict(signed_gap=gap, normal=normal, gradient=normal, owner_id=owner,
                finite_valid=finite, valid=finite, points=q, face_id=face,
                penetration_limit_exceeded=exceeded, inside_closed_owner=any_inside,
                distance_capped=(~active & ~any_inside)))
            if owner_terms:
                si, oi = np.nonzero(np.isfinite(distances) & (inside | (distances < clearance)))
                dist = distances[si, oi]
                sign = np.where(inside[si, oi], -1., 1.)
                pair_key = si*owner_count+oi
                pair_normal = np.zeros((len(si), 3))
                nonzero_pair = dist > 1e-13
                pair_normal[nonzero_pair] = sign[nonzero_pair, None]*(x[si[nonzero_pair]]-witness[pair_key[nonzero_pair]])/dist[nonzero_pair, None]
                pair_normal[~nonzero_pair] = self.face_normals[winface[pair_key[~nonzero_pair]]]
                rows[-1].update(contact_site_index=si+start, contact_owner_id=self.owner_ids[oi],
                    contact_signed_gap=sign*dist, contact_normal=pair_normal)
        if not rows:
            empty = self.query(p)
            empty.update(penetration_limit_exceeded=np.empty(0, bool), inside_closed_owner=np.empty(0, bool), distance_capped=np.empty(0, bool))
            if owner_terms:
                empty.update(contact_site_index=np.empty(0,int), contact_owner_id=np.empty(0,int), contact_signed_gap=np.empty(0), contact_normal=np.empty((0,3)))
            return empty
        return {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}

    def query(self, points, chunk=2048):
        p = self._points(points)
        if not len(p):
            return dict(signed_gap=np.empty(0), normal=np.empty((0, 3)), gradient=np.empty((0, 3)),
                        owner_id=np.empty(0, int), finite_valid=np.empty(0, bool), valid=np.empty(0, bool),
                        points=np.empty((0, 3)), face_id=np.empty(0, int))
        rows = []
        for start in range(0, len(p), chunk):
            x = p[start:start+chunk]
            q, face, _, distance = closest_surface(x, self.vertices, self.faces,
                candidates=self.candidate_count, surface_index=self.index)
            owner = self.face_owner[face].copy()
            inside, valid = self.containment(x, chunk=chunk)
            sign = np.ones(len(x))
            # Any containing owner takes precedence over a nearer unrelated exterior sheet.
            for slot in np.flatnonzero(np.any(inside, axis=0)):
                pid = int(self.owner_ids[slot]); selected = np.flatnonzero(inside[:, slot])
                obj = self.owners[pid]
                if obj['index'] is None:
                    obj['index'] = SurfaceIndex(self.vertices, self.faces[obj['ids']])
                qq, ff, _, dd = closest_surface(x[selected], self.vertices, self.faces[obj['ids']],
                    candidates=self.candidate_count, surface_index=obj['index'])
                choose = (sign[selected] > 0) | (dd > distance[selected])
                idx = selected[choose]
                q[idx], distance[idx], face[idx] = qq[choose], dd[choose], obj['ids'][ff[choose]]
                sign[idx], owner[idx] = -1., pid
            normal = sign[:, None]*(x-q)/np.maximum(distance[:, None], np.finfo(float).tiny)
            on = distance <= 1e-13
            normal[on] = self.face_normals[face[on]]
            finite = valid & np.isfinite(distance) & np.isfinite(normal).all(1)
            rows.append(dict(signed_gap=sign*distance, normal=normal, gradient=normal,
                owner_id=owner, finite_valid=finite, valid=finite, points=q, face_id=face))
        return {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}


def bounded_fixture():
    """Small closed tetrahedron + finite open triangle; not a whole-atlas audit."""
    v = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2],
                  [5, 0, 0], [6, 0, 0], [5, 1, 0]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [4, 5, 6]])
    model = NativeRigidQuery(v, f, [1, 1, 1, 1, 2])
    p = np.array([[.2, .3, .4], [3., .2, .3], [5.2, .2, .4], [5.2, .2, -.4], [7., 0., 0.]])
    t = time.perf_counter(); r = model.query(p)
    assert r['finite_valid'].all()
    assert r['signed_gap'][0] < 0 and np.all(r['signed_gap'][1:] > 0)
    assert r['owner_id'][0] == 1 and np.all(r['owner_id'][2:] == 2)
    assert np.allclose(r['signed_gap'][2:4], [.4, .4])
    assert np.allclose(r['normal'][2:4], [[0, 0, 1], [0, 0, -1]])
    assert np.all(model.distance_lower_bound(p) <= abs(r['signed_gap'])+1e-12)
    return dict(passed=True, points=len(p), seconds=time.perf_counter()-t,
                signed_gaps=r['signed_gap'].tolist(), scope='Bounded synthetic closed-solid/open-sheet test')


def bounded_band_fixture():
    """Reproducible bounded band continuity, occupancy and finite-sheet checks."""
    v = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2],
                  [5, 0, 0], [6, 0, 0], [5, 1, 0]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [4, 5, 6]])
    model = NativeRigidQuery(v, f, [1, 1, 1, 1, 2])
    p = np.array([[.02, .3, .4], [-.02, .3, .4], [.2, .3, .4],
                  [5.2, .2, .04], [5.2, .2, -.04], [8., 0, 0]])
    old, new = model.query(p), model.query_band(p)
    active = [0, 1, 3, 4]
    assert np.allclose(new['signed_gap'][active], old['signed_gap'][active])
    assert np.allclose(new['normal'][active], old['normal'][active])
    assert not new['finite_valid'][2] and new['penetration_limit_exceeded'][2]
    assert new['signed_gap'][5] == .1 and np.all(new['normal'][5] == 0)
    error = 0.
    for k in active:
        for j in range(3):
            plus, minus = p[[k]].copy(), p[[k]].copy()
            plus[0, j] += 1e-6; minus[0, j] -= 1e-6
            derivative = (model.query_band(plus)['signed_gap'][0]-model.query_band(minus)['signed_gap'][0])/2e-6
            error = max(error, abs(derivative-new['normal'][k, j]))
    assert error < 1e-7
    return dict(passed=True, points=len(p), maximum_gradient_fd_error=float(error),
                missing_containing_owner_witness_rejected=True,
                outside_band_zero_normal=True, scope='Bounded synthetic point-site query-band fixture')


if __name__ == '__main__':
    print(json.dumps(dict(basic=bounded_fixture(), band=bounded_band_fixture()), indent=2))
