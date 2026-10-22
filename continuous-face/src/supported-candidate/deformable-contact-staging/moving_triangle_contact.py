"""Isolated reciprocal finite moving-sheet contact, not a full model adapter.

Both tissue samples and source vertices are convex carriers of current global
DOFs. The barrier uses finite UNSIGNED point/triangle distance. Anatomical side
qualification is a separate, explicit reference diagnostic, never a signed
infinite-plane force or a claim of closed-volume inside/outside classification.

All explicitly eligible pairs are evaluated, including inactive distant pairs.
The affine-path bound uses a fresh separating direction for every finite pair,
with outward-rounded interval carrier/projection arithmetic. It is not mesh
CCD: unsampled tissue, edge/edge collisions, omitted pairs and nonlinear paths
are outside its guarantee. Caller must also retain current endpoint, material,
energy and convergence acceptance checks.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags


class InfeasibleContactError(ValueError):
    pass


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _iadd(a, b):
    return _down(a[0]+b[0]), _up(a[1]+b[1])


def _isub(a, b):
    return _down(a[0]-b[1]), _up(a[1]-b[0])


def _imul(a, b):
    values = np.stack((a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    return _down(values.min(axis=0)), _up(values.max(axis=0))


def _idot(a, b):
    product = _imul(a, b)
    out = product[0][..., 0], product[1][..., 0]
    for j in (1, 2):
        out = _iadd(out, (product[0][..., j], product[1][..., j]))
    return out


def _map_interval_scalar(matrix, x):
    """Enclose the exact real-valued product of represented CSR and x.

    Separate products and additions, each rounded outwards. No BLAS error
    estimate, fused arithmetic, or unguarded sparse accumulation is assumed.
    This deliberately simple staging implementation favors auditability.
    """
    low, high = np.zeros((matrix.shape[0], 3)), np.zeros((matrix.shape[0], 3))
    for row in range(matrix.shape[0]):
        total = np.zeros(3), np.zeros(3)
        for k in range(matrix.indptr[row], matrix.indptr[row+1]):
            a, v = matrix.data[k], x[matrix.indices[k]]
            total = _iadd(total, _imul((a, a), (v, v)))
        low[row], high[row] = total
    return low, high


def _map_interval(matrix, x):
    """Enclose the represented CSR product, preserving each row's exact order.

    Batch independent finite float64 CSR rows at the same stored coefficient
    position. Retain every _imul/_iadd operation and outward rounding; do
    not canonicalize, eliminate zeros, reorder, fuse, or use sparse BLAS.
    Unsupported inputs retain the original scalar/array promotion behavior.
    Nonfinite inputs also fall back to preserve scalar NaN payload selection.
    """
    if (type(matrix) is not csr_matrix or type(matrix.data) is not np.ndarray
            or matrix.data.dtype != np.dtype(np.float64)
            or type(x) is not np.ndarray or x.dtype != np.dtype(np.float64)
            or x.shape != (matrix.shape[1], 3)
            or not np.isfinite(matrix.data).all() or not np.isfinite(x).all()):
        return _map_interval_scalar(matrix, x)

    low, high = np.zeros((matrix.shape[0], 3)), np.zeros((matrix.shape[0], 3))
    starts = matrix.indptr[:-1]
    lengths = np.diff(matrix.indptr)
    active = np.flatnonzero(lengths)
    position = 0
    while active.size:
        k = starts[active] + position
        a, v = matrix.data[k, None], x[matrix.indices[k]]
        total = _iadd((low[active], high[active]), _imul((a, a), (v, v)))
        low[active], high[active] = total
        position += 1
        active = active[lengths[active] > position]
    return low, high


def activated_log_barrier(gap, clearance):
    """b(g)=-(g-h)^2 log(g/h) for 0<g<h, otherwise zero or infeasible.

    Returned first and second derivatives are with respect to finite gap.
    """
    h = float(clearance)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("positive finite activation clearance required")
    g = np.asarray(gap, float)
    values = [np.zeros_like(g) for _ in range(3)]
    invalid = ~np.isfinite(g) | (g <= 0)
    values[0][invalid], values[1][invalid], values[2][invalid] = np.inf, np.nan, np.nan
    active = ~invalid & (g < h)
    x, u = g[active], h-g[active]
    near = x > .5*h
    logratio = np.empty_like(x)
    logratio[near] = np.log1p((x[near]-h)/h)
    logratio[~near] = np.log(x[~near])-np.log(h)
    with np.errstate(over="ignore", divide="ignore"):
        ratio = u/x
        values[0][active] = -u*u*logratio
        values[1][active] = 2*u*logratio-u*ratio
        values[2][active] = -2*logratio+4*ratio+ratio*ratio
    return tuple(values)


def point_triangle_geometry(points, triangles, *, rank_tolerance=1e-12):
    """Paired finite triangle closest points with exact-branch distance gradient.

    Return barycentric carrier and normal r/|r|. At nonzero gap the envelope
    derivative is [n, -b0*n, -b1*n, -b2*n]; differentiating the barycentrics
    again is NOT required for this first derivative. Degenerate triangles are
    rejected. Boundary features are included, not extrapolated face planes.
    """
    p, tri = np.asarray(points, float), np.asarray(triangles, float)
    if (p.ndim != 2 or p.shape[1] != 3 or tri.shape != (len(p), 3, 3)
            or not np.isfinite(p).all() or not np.isfinite(tri).all()):
        raise ValueError("finite paired N x 3 points and N x 3 x 3 triangles required")
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("positive finite triangle rank tolerance required")
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b-a, c-a
    scale = np.maximum(np.linalg.norm(ab, axis=1), np.linalg.norm(ac, axis=1))
    if np.any(~np.isfinite(scale) | (scale <= 0)):
        raise InfeasibleContactError("degenerate or unrepresentable source triangle")
    uedge, vedge = ab/scale[:, None], ac/scale[:, None]
    face_cross = np.cross(uedge, vedge)
    rank = np.linalg.norm(face_cross, axis=1)
    if np.any(~np.isfinite(rank) | (rank <= rank_tolerance)):
        raise InfeasibleContactError("unresolved or degenerate source triangle")
    face_normal = face_cross/rank[:, None]
    ap = (p-a)/scale[:, None]
    denominator = rank*rank
    u = np.einsum("ij,ij->i", np.cross(ap, vedge), face_cross)/denominator
    v = np.einsum("ij,ij->i", np.cross(uedge, ap), face_cross)/denominator
    projected_bary = np.column_stack((1-u-v, u, v))
    bary = projected_bary.copy()
    closest = a+u[:, None]*ab+v[:, None]*ac
    d2 = np.sum((p-closest)**2, axis=1)
    d2[(bary < 0).any(axis=1)] = np.inf
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = tri[:, j]-tri[:, i]
        fraction = np.clip(np.einsum("ij,ij->i", p-tri[:, i], edge)/
                           np.einsum("ij,ij->i", edge, edge), 0., 1.)
        foot = tri[:, i]+fraction[:, None]*edge
        candidate = np.sum((p-foot)**2, axis=1)
        choose = candidate < d2
        edge_bary = np.zeros_like(bary)
        edge_bary[:, i], edge_bary[:, j] = 1-fraction, fraction
        closest[choose], d2[choose], bary[choose] = foot[choose], candidate[choose], edge_bary[choose]
    gap = np.sqrt(d2)
    delta = p-closest
    normal = np.divide(delta, gap[:, None], out=np.zeros_like(delta), where=gap[:, None] > 0)
    # This is a side DIAGNOSTIC. It never replaces finite unsigned gap.
    oriented_height = np.einsum("ij,ij->i", face_normal, p-a)
    local_gradient = np.concatenate((normal[:, None], -bary[:, :, None]*normal[:, None]), axis=1)
    return dict(closest=closest, gap=gap, barycentric=bary, normal=normal,
                local_gradient=local_gradient, face_normal=face_normal,
                oriented_height=oriented_height, projected_barycentric=projected_bary,
                scaled_triangle_rank=rank,
                feature_dimension=(bary > 128*np.finfo(float).eps).sum(axis=1)-1)


def _carrier(matrix, name):
    C = csr_matrix(matrix, dtype=float, copy=True)
    C.sum_duplicates()
    C.eliminate_zeros()
    row_sum = np.asarray(C.sum(axis=1)).ravel()
    if (not np.isfinite(C.data).all() or np.any(C.data < 0) or
            np.any(abs(row_sum-1) > 64*np.finfo(float).eps)):
        raise ValueError(f"{name} must contain finite nonnegative partition-of-unity rows")
    for a in (C.data, C.indices, C.indptr):
        a.flags.writeable = False
    return C


@dataclass(frozen=True)
class MovingContactState:
    points: np.ndarray
    vertices: np.ndarray
    geometry: dict
    support_lower: np.ndarray
    feasible: bool
    reason: str


@dataclass(frozen=True)
class MovingPathReport:
    step_bound: float
    max_step: float
    eligible_pairs: int
    limiting_pair: int | None
    minimum_support_lower: float
    interval_guarded_carriers: bool = True
    guarantee: str = "selected point/finite-triangle affine trajectories only"


class MovingTriangleContact:
    """Reciprocal moving-sheet barrier with explicit fixed eligibility.

    C maps global nodes q to tissue samples. M maps the same q to source mesh
    vertices. faces indexes M rows. pairs is an integer (P,2) [sample,face]
    list OR eligibility is a Boolean (num_samples,num_faces) mask, not both.
    False/missing pairs, including anatomically bonded neighbors, are excluded
    by the caller. Pair weights are fixed reference quadrature/stiffness
    factors, never deformation-dependent area or an outer-face target.

    Every eligible pair contributes; this is not nearest-face-only contact.
    Near shared edges this sum is mesh/pair-weight dependent. No broadphase,
    self-contact coverage, or area-complete triangle contact is claimed.
    """
    def __init__(self, point_carrier, vertex_carrier, faces, *, pairs=None,
                 eligibility=None, pair_weights=1., clearance=.02, rank_tolerance=1e-12):
        self.C, self.M = _carrier(point_carrier, "point carrier"), _carrier(vertex_carrier, "vertex carrier")
        if self.C.shape[1] != self.M.shape[1]:
            raise ValueError("point and moving source carriers must use the same global DOFs")
        raw = np.asarray(faces)
        if (raw.ndim != 2 or raw.shape[1] != 3 or not np.issubdtype(raw.dtype, np.integer)
                or not len(raw) or raw.min() < 0 or raw.max() >= self.M.shape[0]
                or np.any(np.diff(np.sort(raw, axis=1), axis=1) == 0)):
            raise ValueError("nonempty integer faces with three distinct in-range source vertices required")
        self.faces = np.array(raw, dtype=np.int64, copy=True)
        if (pairs is None) == (eligibility is None):
            raise ValueError("provide exactly one explicit pair list or eligibility mask")
        if eligibility is not None:
            eligible = np.asarray(eligibility)
            if eligible.dtype != bool or eligible.shape != (self.C.shape[0], len(self.faces)):
                raise ValueError("Boolean sample-by-face eligibility required")
            pairs = np.argwhere(eligible)
        pp = np.asarray(pairs)
        if (pp.ndim != 2 or pp.shape[1] != 2 or not np.issubdtype(pp.dtype, np.integer)
                or not len(pp) or np.any(pp < 0) or np.any(pp[:, 0] >= self.C.shape[0])
                or np.any(pp[:, 1] >= len(self.faces)) or len(np.unique(pp, axis=0)) != len(pp)):
            raise ValueError("unique nonempty in-range integer [sample,face] pairs required")
        self.pairs = np.array(pp, dtype=np.int64, copy=True)
        self.weights = np.broadcast_to(np.asarray(pair_weights, float), (len(pp),)).copy()
        self.clearance, self.rank_tolerance = float(clearance), float(rank_tolerance)
        if (not np.isfinite(self.weights).all() or np.any(self.weights <= 0) or
                not np.isfinite(self.clearance) or self.clearance <= 0 or
                not np.isfinite(self.rank_tolerance) or self.rank_tolerance <= 0):
            raise ValueError("positive finite weights, clearance, and rank tolerance required")
        self._face_nodes = self.faces[self.pairs[:, 1]]
        for a in (self.faces, self.pairs, self.weights, self._face_nodes):
            a.flags.writeable = False

    def _q(self, q):
        x = np.asarray(q, float)
        if x.shape != (self.C.shape[1], 3) or not np.isfinite(x).all():
            raise ValueError("finite global node_count x 3 positions or direction required")
        return x

    def _relative_intervals(self, x):
        P, V = _map_interval(self.C, x), _map_interval(self.M, x)
        return _isub(tuple(a[self.pairs[:, 0], None, :] for a in P),
                     tuple(a[self._face_nodes] for a in V))

    def evaluate(self, q):
        q = self._q(q)
        points, vertices = np.asarray(self.C@q), np.asarray(self.M@q)
        try:
            geometry = point_triangle_geometry(points[self.pairs[:, 0]], vertices[self._face_nodes],
                                               rank_tolerance=self.rank_tolerance)
        except InfeasibleContactError as exc:
            return MovingContactState(points, vertices, dict(gap=np.full(len(self.pairs), np.nan)),
                                      np.zeros(len(self.pairs)), False, str(exc))
        # Any direction works for a separation certificate, so numerical
        # closest-point error cannot silently invalidate this lower bound.
        n = geometry['normal'][:, None, :]
        projection = _idot((n, n), self._relative_intervals(q))
        support_lower = projection[0].min(axis=1)
        feasible = (np.isfinite(support_lower) & (support_lower > 0) &
                    np.isfinite(geometry['gap']) & (geometry['gap'] > 0))
        reason = "" if feasible.all() else f"{int((~feasible).sum())} touching or numerically unresolved finite pairs"
        return MovingContactState(points, vertices, geometry, support_lower, bool(feasible.all()), reason)

    def require_feasible(self, q):
        state = self.evaluate(q)
        if not state.feasible:
            raise InfeasibleContactError(state.reason)
        return state

    def __call__(self, q):
        state = self.require_feasible(q)
        b, db, _ = activated_log_barrier(state.geometry['gap'], self.clearance)
        force_factor = self.weights*db
        local = force_factor[:, None, None]*state.geometry['local_gradient']
        gp, gv = np.zeros_like(state.points), np.zeros_like(state.vertices)
        np.add.at(gp, self.pairs[:, 0], local[:, 0])
        for k in range(3):
            np.add.at(gv, self._face_nodes[:, k], local[:, k+1])
        return float(self.weights@b), np.asarray(self.C.T@gp+self.M.T@gv)

    def _gap_jacobian(self, state):
        if not state.feasible:
            raise InfeasibleContactError(state.reason)
        geom = state.geometry
        carrier = self.C[self.pairs[:, 0]].copy()
        for k in range(3):
            carrier -= diags(geom['barycentric'][:, k])@self.M[self._face_nodes[:, k]]
        carrier = carrier.tocoo()
        normal = geom['normal'][carrier.row]
        return coo_matrix(((carrier.data[:, None]*normal).ravel(),
                           (np.repeat(carrier.row, 3), (3*carrier.col[:, None]+np.arange(3)).ravel())),
                          shape=(len(self.pairs), 3*self.C.shape[1])).tocsr()

    def gap_jacobian(self, q):
        return self._gap_jacobian(self.require_feasible(q))

    def normal_tangent(self, q):
        """PSD b''(d) J_d^T J_d, not the full derivative of contact force.

        Moving normals/barycentrics add b'(d) Hessian(d) to the true Hessian.
        They are intentionally omitted from this solver tangent, not from the
        first derivative. Rotation invariance does not imply finite-rotation
        nullspace for the full Hessian at a nonstationary configuration.
        """
        state = self.require_feasible(q)
        J = self._gap_jacobian(state)
        curvature = activated_log_barrier(state.geometry['gap'], self.clearance)[2]
        return (J.T@diags(self.weights*curvature)@J).tocsr()

    def path_bound(self, q, direction, *, safety=.9, max_step=1.):
        """Prevent selected point/triangle intersection along an affine step.

        With a freshly chosen fixed separating n, every triangle vertex has
        n dot (p(t)-a_i(t)) >= L_i + t V_i. Positivity for all vertices means
        positivity for every finite barycentric triangle point. Outward
        intervals enclose L_i and V_i directly from both moving carriers.
        No endpoint-only feasibility assumption and no active-pair truncation.

        This certificate does not ensure positive triangle area or material J
        along the path. Those separate constraints belong to the solid solver.
        Float64 IEEE-754, finite intermediate arithmetic is required; overflow
        or unresolved separation fails closed. No nonlinear trajectory claim.
        """
        if not 0 < safety < 1 or not np.isfinite(max_step) or max_step < 0:
            raise ValueError("need 0<safety<1 and finite max_step>=0")
        state, direction = self.require_feasible(q), self._q(direction)
        n = state.geometry['normal'][:, None, :]
        initial = _idot((n, n), self._relative_intervals(self._q(q)))[0]
        velocity = _idot((n, n), self._relative_intervals(direction))[0]
        if not np.isfinite(initial).all() or not np.isfinite(velocity).all():
            raise InfeasibleContactError("nonfinite interval path arithmetic")
        decreasing = velocity < 0
        pair_limits = np.full_like(initial, np.inf)
        with np.errstate(over='ignore', divide='ignore'):
            pair_limits[decreasing] = _down(safety*_down(initial[decreasing]/(-velocity[decreasing])))
        limits = pair_limits.min(axis=1)
        best = int(np.argmin(limits))
        limit = max(0., min(float(max_step), float(limits[best])))
        limiting = best if limits[best] < max_step else None
        return MovingPathReport(limit, float(max_step), len(self.pairs), limiting,
                                float(state.support_lower.min()))

    def safe_step_bound(self, q, direction, *, safety=.9, max_step=1.):
        return self.path_bound(q, direction, safety=safety, max_step=max_step).step_bound

    def check_trial(self, q, direction, alpha, *, safety=.9):
        """Contact-only acceptance gate, not energy/material/convergence approval."""
        if not np.isfinite(alpha) or alpha < 0:
            return False, "nonnegative finite step required"
        report = self.path_bound(q, direction, safety=safety, max_step=float(alpha))
        if alpha > report.step_bound:
            return False, "affine path is not covered by the finite-pair separation certificate"
        try:
            self.require_feasible(self._q(q)+alpha*self._q(direction))
        except InfeasibleContactError as exc:
            return False, f"current endpoint rejected: {exc}"
        return True, "contact path and current endpoint only; solver acceptance still required"

    def validate_selected_sides(self, q, sample_ids, face_ids, expected_sides=1., *, tolerance=1e-10):
        """Qualify explicit anatomy-selected reference interface orientations.

        Side is the oriented CURRENT triangle plane diagnostic, reported with
        finite foot/feature information. It is never fed into the force, used
        for arbitrary distant faces, or interpreted as closed-volume sign.
        Negative/ambiguous orientation must be resolved by anatomy assembly,
        not by silently flipping normals or freezing the muscle. Open sheets
        allow going around a boundary to the opposite side without collision.
        """
        q = self._q(q)
        s, f = np.asarray(sample_ids), np.asarray(face_ids)
        if (s.ndim != 1 or f.shape != s.shape or not len(s) or
                not np.issubdtype(s.dtype, np.integer) or not np.issubdtype(f.dtype, np.integer)
                or np.any(s < 0) or np.any(s >= self.C.shape[0]) or np.any(f < 0) or np.any(f >= len(self.faces))):
            raise ValueError("paired in-range integer side qualification indices required")
        side = np.broadcast_to(np.asarray(expected_sides, float), s.shape)
        if np.any(abs(side) != 1) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("expected sides must be +/-1 and tolerance nonnegative finite")
        P, V = np.asarray(self.C@q)[s], np.asarray(self.M@q)[self.faces[f]]
        geom = point_triangle_geometry(P, V, rank_tolerance=self.rank_tolerance)
        margin = side*geom['oriented_height']
        return dict(valid=bool(np.all(margin > tolerance)), side_margin=margin,
                    expected_sides=side.copy(), finite_distance=geom['gap'],
                    projected_inside_face=np.all(geom['projected_barycentric'] >= 0, axis=1),
                    closest_feature_dimension=geom['feature_dimension'],
                    closed_volume_sign_supplied=False, used_in_barrier=False)
