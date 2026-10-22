"""Audited pure helper subset extracted from archived source. No archive main or target coordinates."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_closing,binary_fill_holes,binary_dilation,label

def projected_aperture(vertices, faces, axes=(0, 1), spacing=0.5):
    """Largest actual finite projected-surface hole; no hand-shaped ellipse."""
    tri = vertices[faces][:, :, axes]
    low = np.floor((tri.min((0, 1)) - 2) / spacing) * spacing
    high = np.ceil((tri.max((0, 1)) + 2) / spacing) * spacing
    shape = np.rint((high - low) / spacing).astype(int) + 1
    idx = np.indices(shape).reshape(2, -1).T
    p = low + spacing * idx
    c = tri.mean(1)
    r = np.linalg.norm(tri - c[:, None], axis=2).max(1)
    tree = cKDTree(c)
    hit = np.zeros(len(p), bool)
    for start in range(0, len(p), 1024):
        q = p[start:start + 1024]
        choices = tree.query_ball_point(q, float(r.max()) + 1e-09)
        count = np.fromiter(map(len, choices), int, len(q))
        pi = np.repeat(np.arange(len(q)), count)
        fi = np.concatenate(choices).astype(int) if count.sum() else np.empty(0, int)
        d = q[pi] - tri[fi, 0]
        a = tri[fi, 1] - tri[fi, 0]
        b = tri[fi, 2] - tri[fi, 0]
        det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        safe = np.where(abs(det) > 1e-12, det, 1.0)
        u = (d[:, 0] * b[:, 1] - d[:, 1] * b[:, 0]) / safe
        v = (a[:, 0] * d[:, 1] - a[:, 1] * d[:, 0]) / safe
        yes = (abs(det) > 1e-12) & (u >= -1e-09) & (v >= -1e-09) & (u + v <= 1 + 1e-09)
        hit[start + np.unique(pi[yes])] = True
    covered = hit.reshape(tuple(shape))
    closed = binary_closing(covered, iterations=1)
    holes = binary_fill_holes(closed) & ~closed
    components, n = label(holes)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    if n == 0:
        raise ValueError('Source has no projected aperture')
    aperture = components == int(sizes.argmax())
    points = p[aperture.ravel()]
    return (dict(origin=low, spacing=np.array(spacing), mask=aperture, source_coverage=covered), dict(area=float(aperture.sum() * spacing ** 2), bounds=[points.min(0).tolist(), points.max(0).tolist()], source_faces=len(faces), sample_spacing=spacing, authored_projection_seam_closing_radius=spacing, discarded_small_projected_holes=int(n - 1)))

def aperture_at(p, a, padding=0.0):
    mask = a['mask']
    if padding:
        mask = binary_dilation(mask, iterations=int(np.ceil(padding / float(a['spacing']))))
    ij = np.rint((p[:, :2] - a['origin']) / float(a['spacing'])).astype(int)
    valid = np.all((ij >= 0) & (ij < np.array(mask.shape)), axis=1)
    inside = np.zeros(len(p), bool)
    inside[valid] = mask[tuple(ij[valid].T)]
    return inside

def boundary_from_cells(occupancy, origin, spacing):
    faces = []
    owner = []
    for axis in range(3):
        for sign in (-1, 1):
            neighbour = np.roll(occupancy, -sign, axis=axis)
            sl = [slice(None)] * 3
            sl[axis] = -1 if sign == 1 else 0
            neighbour[tuple(sl)] = False
            ids = np.argwhere(occupancy & ~neighbour)
            base = ids.copy()
            base[:, axis] += sign == 1
            other = [a for a in range(3) if a != axis]
            corners = np.repeat(base[:, None, :], 4, axis=1)
            corners[:, 1, other[0]] += 1
            corners[:, 2, other] += 1
            corners[:, 3, other[1]] += 1
            normal = np.cross(corners[0, 1] - corners[0, 0], corners[0, 2] - corners[0, 0]) if len(corners) else np.zeros(3)
            if normal[axis] * sign < 0:
                corners = corners[:, [0, 3, 2, 1]]
            faces.append(corners[:, [0, 1, 2]])
            faces.append(corners[:, [0, 2, 3]])
            owner.extend([ids, ids])
    integer = np.concatenate(faces)
    unique, inverse = np.unique(integer.reshape(-1, 3), axis=0, return_inverse=True)
    return (origin + spacing * unique, inverse.reshape(-1, 3), np.concatenate(owner))
