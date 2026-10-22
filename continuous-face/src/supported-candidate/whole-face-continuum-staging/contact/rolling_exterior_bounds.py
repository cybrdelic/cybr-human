"""ISOLATED prototype: conservative site screening, not a live guard adapter.

An anchor is admitted only after query_band independently establishes exterior
membership for every signed closed owner. The refreshed distance lower bound
is to ALL finite native facets, including unsigned sheets. Its surface-free
ball preserves closed-owner membership. No active force or guard is removed.
"""
import hashlib
import time
import numpy as np


def geometry_token(rigid):
    h = hashlib.sha256()
    for a in (rigid.vertices, rigid.faces, rigid.face_owner):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr([(int(k), bool(v['closed'])) for k, v in rigid.owners.items()]).encode())
    return h.hexdigest()


class RollingExteriorBounds:
    def __init__(self, rigid, reference_points, reference_lower, clearance=.1, refresh_radius=.5):
        self.rigid = rigid
        self.token = geometry_token(rigid)
        self.reference = np.asarray(reference_points, float).copy()
        self.reference_lower = np.asarray(reference_lower, float).copy()
        if (self.reference.ndim != 2 or self.reference.shape[1] != 3
                or not np.isfinite(self.reference).all()
                or self.reference_lower.shape != (len(self.reference),)
                or not np.isfinite(self.reference_lower).all()):
            raise ValueError('Finite reference points and per-site lower bounds required')
        self.h = float(clearance)
        self.refresh_radius = float(refresh_radius)
        if not np.isfinite([self.h, self.refresh_radius]).all() or not 0 < self.h < self.refresh_radius:
            raise ValueError('Numerical refresh radius must exceed unchanged physical clearance')
        self.anchors = self.reference.copy()
        self.anchor_lower = np.zeros(len(self.reference))
        self.eligible = np.zeros(len(self.reference), bool)

    def _check_geometry(self):
        if geometry_token(self.rigid) != self.token:
            raise ValueError('Native geometry or signed-owner semantics changed; anchors invalid')

    def _points(self, points):
        p = np.asarray(points, float)
        if p.shape != self.reference.shape or not np.isfinite(p).all():
            raise ValueError('Finite points with the exact reference shape required')
        return p

    def lower_bound(self, points):
        self._check_geometry()
        p = self._points(points)
        # Additional subtraction makes the Lipschitz translation conservative
        # under the same padded floating-point model as the existing queries.
        pad = 1024*np.finfo(float).eps*(np.linalg.norm(p, axis=1)
              + np.linalg.norm(self.anchors, axis=1) + self.refresh_radius + 1)
        reference_pad = 1024*np.finfo(float).eps*(np.linalg.norm(p, axis=1)
                        + np.linalg.norm(self.reference, axis=1) + 1)
        original = self.reference_lower-np.linalg.norm(p-self.reference, axis=1)-reference_pad
        rolling = self.anchor_lower-np.linalg.norm(p-self.anchors, axis=1)-pad
        rolling[~self.eligible] = -np.inf
        return np.maximum(original, rolling)

    def refresh(self, points, queried_ids, result):
        self._check_geometry()
        started = time.perf_counter()
        ids = np.asarray(queried_ids, int)
        p = self._points(points)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(p)):
            raise ValueError('Valid per-site query indices required')
        for key in ('valid', 'inside_closed_owner', 'distance_capped'):
            value = np.asarray(result[key])
            if value.shape != ids.shape or value.dtype != np.dtype(bool):
                raise ValueError('Aligned boolean all-owner query evidence required')
        # valid alone also admits shallow interior points. Require BOTH fields.
        accepted = (np.asarray(result['valid'], bool)
                    & ~np.asarray(result['inside_closed_owner'], bool)
                    & np.asarray(result['distance_capped'], bool))
        selected = ids[accepted]
        lower = self.rigid.clipped_distance_lower_bound(p[selected], self.refresh_radius)
        useful = lower > self.h
        self.anchors[selected[useful]] = p[selected[useful]]
        self.anchor_lower[selected[useful]] = lower[useful]
        self.eligible[selected[useful]] = True
        return dict(eligible_exterior_inactive_sites=int(len(selected)),
                    stored_useful_anchors=int(useful.sum()),
                    rejected_inside_sites=int(np.sum(result['inside_closed_owner'])),
                    refresh_seconds=time.perf_counter()-started,
                    numerical_refresh_radius_mm=self.refresh_radius)
