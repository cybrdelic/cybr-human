"""Code-authored external feature geometry, guided by cited human anatomy.

Dimensions here author one adult morphology; they are not universal norms or
an assertion of biomechanical calibration. Native coordinates are millimetres.
"""

import numpy as np

REFERENCES = {
    "nose_profile": "https://onlinelibrary.wiley.com/doi/10.1111/j.1469-7580.2008.00924.x",
    "nasal_envelope": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6735417/",
    "alar_support": "https://pubmed.ncbi.nlm.nih.gov/32842101/",
    "ear_landmarks": "https://onlinelibrary.wiley.com/doi/full/10.1002/ajmg.a.32599",
    "ear_mechanics": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4512857/",
    "eyelid_imaging": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7139934/",
    "limbus_geometry": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10842492/",
}


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))


def smooth_minimum(a, b, width=0.3):
    d = abs(a - b)
    t = np.minimum(d / width, 1)
    rounded = width * (0.375 + 0.75 * t * t - 0.125 * t**4)
    return np.where(d < width, 0.5 * (a + b - rounded), np.minimum(a, b))


def pinna_height(u, z, r):
    g = lambda cu, cz, su, sz: np.exp(
        -0.5 * ((u - cu) / su) ** 2 - 0.5 * ((z - cz) / sz) ** 2
    )
    h = 5.1 + 4.2 * np.exp(-(((r - 0.91) / 0.045) ** 2)) * smooth(-23, -12, z)
    h -= 3.3 * g(-2, -2, 5.2, 11)
    h += 2.6 * g(3, 0, 2.0, 13)
    h += (
        1.7
        * np.exp(-(((u - 3 - 0.23 * np.maximum(z, 0)) / 1.8) ** 2))
        * smooth(1, 7, z)
        * (1 - smooth(20, 27, z))
    )
    h += (
        1.6
        * np.exp(-(((z - 10 + 0.65 * (u - 2)) / 1.8) ** 2))
        * smooth(0, 3, u)
        * (1 - smooth(9, 13, u))
    )
    h += 3.1 * g(-7, -4, 1.8, 4.0) + 1.7 * g(-1, -13, 2.6, 3.2)
    h -= 1.2 * g(-4, -10, 1.6, 2.0)
    return h + 2 * g(0, -23, 6, 5)


def pinna_chart_radius(u, z):
    a = (u - 1.3 * z / 30) / 13.8
    b = z / 29.5
    r = np.maximum(np.hypot(a, b), 0.001)
    for _ in range(12):
        sin = b / r
        den = r + 0.13 * b
        res = (a / den) ** 2 + sin * sin - 1
        derivative = -2 * a * a / den**3 - 2 * b * b / r**3
        r = np.maximum(0.001, r - res / np.minimum(derivative, -1e-12))
    return r


def attached_pinna(vertices, faces, side, centre):
    """Free helical rim, posterior surface and a shared actual scalp root.

    Remove a localized root patch, rather than pinning the ear perimeter to the
    head. Every new root point is an existing head boundary vertex.
    """
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int32)
    centre = np.asarray(centre, float)
    normal = np.array([side * 0.94, -0.3412, 0.0])
    normal /= np.linalg.norm(normal)
    horizontal = np.array([side * 0.3412, 0.94, 0.0])
    horizontal /= np.linalg.norm(horizontal)
    mid = vertices[faces].mean(1)
    relative = mid - centre
    u = relative @ horizontal
    zz = relative[:, 2]
    remove = (
        ((u / 5.4) ** 2 + (zz / 17) ** 2 < 1)
        & (mid[:, 0] * side > centre[0] * side - 15)
        & (abs(relative @ normal) < 8)
    )
    if remove.sum() < 20:
        raise ValueError("Missing auricular head attachment patch")
    # Weld geometric UV aliases to recover the actual boundary graph.
    ids = np.unique(faces[remove])
    keys = np.rint(vertices[ids] * 1e5).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    lookup = np.full(len(vertices), -1, np.int32)
    lookup[ids] = inverse
    f = lookup[faces[remove]]
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges.sort(1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    edge = edges[counts == 1]
    graph = {}
    for a, b in edge:
        graph.setdefault(int(a), []).append(int(b))
        graph.setdefault(int(b), []).append(int(a))
    if not graph or any(len(v) != 2 for v in graph.values()):
        raise ValueError("Auricular root is not a closed manifold loop")
    start = next(iter(graph))
    order = [start]
    prev = -1
    current = start
    while True:
        following = next(j for j in graph[current] if j != prev)
        if following == start:
            break
        order.append(following)
        prev, current = current, following
        if len(order) > len(graph):
            raise ValueError("Broken root loop")
    if len(order) != len(graph):
        raise ValueError("Multiple auricular root loops")
    roots = vertices[ids[first[order]]]
    rel = roots - centre
    angle = np.unwrap(np.arctan2(rel[:, 2] / 17, (rel @ horizontal) / 5.4))
    if np.median(np.diff(angle)) < 0:
        roots = roots[::-1]
        rel = roots - centre
        angle = np.unwrap(np.arctan2(rel[:, 2] / 17, (rel @ horizontal) / 5.4))
    chart = np.column_stack([(rel @ horizontal) / 5.4, rel[:, 2] / 17])
    arc = np.linalg.norm(np.diff(np.vstack([chart, chart[:1]]), axis=0), axis=1)
    angle = angle[0] + 2 * np.pi * np.r_[0, np.cumsum(arc[:-1])] / arc.sum()
    # Preserve the graph order and exact scalp roots. Uneven angular sampling
    # is inherited from the actual head chart, not replaced by guessed roots.
    n = len(roots)
    rr = np.linspace(0.025, 1, 48)
    theta = angle[None, :]
    r = rr[:, None]
    u = 13.8 * r * np.cos(theta) * (1 + 0.13 * np.sin(theta))
    z = 29.5 * r * np.sin(theta)
    u += 1.3 * z / 30
    rootlocal = roots - centre
    rootu = rootlocal @ horizontal
    rootz = rootlocal[:, 2]
    rootheight = rootlocal @ normal
    rootfront = pinna_height(rootu, rootz, pinna_chart_radius(rootu, rootz))
    projection = max(0.0, float(np.max(rootheight + 2.7 - rootfront)))
    h = pinna_height(u, z, r) + projection
    front = (
        centre
        + u[..., None] * horizontal
        + z[..., None] * [0, 0, 1]
        + h[..., None] * normal
    )
    # Back surface, joined to the front at a rounded free helical rim. Its
    # central attachment passes through a sulcus before projecting outward.
    back = front.copy()
    back -= normal * (1.15 + 0.9 * smooth(0.7, 1, r))[..., None]
    t = np.linspace(0, 1, 48)[:, None, None]
    mix = smooth(0, 1, t)
    # Expand monotonically from the scalp root to the free rim. Blending a
    # large root into small central disk rings would turn the rear sheet back
    # on itself before it reached the rim.
    back = roots[None] * (1 - mix) + back[-1:] * mix
    local = back - centre
    bu = local @ horizontal
    bz = local[..., 2]
    bh = local @ normal
    canal_region = 1 - smooth(
        1, 1.8, np.sqrt(((bu + 4) / 2.4) ** 2 + ((bz + 5) / 3.4) ** 2)
    )
    limit = (
        pinna_height(bu, bz, pinna_chart_radius(bu, bz))
        + projection
        - 1.15
        - canal_region
    )
    capped = smooth_minimum(bh, limit, 0.3)
    back += normal * (capped - bh)[..., None]
    verts = np.vstack(
        [
            front.reshape(-1, 3),
            back.reshape(-1, 3),
            centre + normal * (pinna_height(0.0, 0.0, 0.0) + projection),
        ]
    )
    fs = []
    for base in [0, 48 * n]:
        for j in range(47):
            for k in range(n):
                a = base + j * n + k
                b = base + j * n + (k + 1) % n
                c = a + n
                dd = b + n
                fs.extend(
                    [(a, c, b), (b, c, dd)] if base == 0 else [(a, b, c), (b, dd, c)]
                )
    pole = len(verts) - 1
    for k in range(n):
        fs.append((pole, k, (k + 1) % n))
        a = 47 * n + k
        b = 47 * n + (k + 1) % n
        c = 48 * n + a
        dd = 48 * n + b
        fs.extend([(a, b, c), (b, dd, c)])
    fs = np.asarray(fs, np.int32)
    # A recessed external meatus entrance, behind the tragus. This represents
    # only the entrance; it does not claim the full internal auditory canal.
    relative = verts - centre
    inside = ((relative @ horizontal + 4) / 2.4) ** 2 + (
        (relative[:, 2] + 5) / 3.4
    ) ** 2 < 1
    cut = inside[fs].any(1)
    cut &= np.all(fs < 48 * n, axis=1)
    ed = np.vstack([fs[cut][:, [0, 1]], fs[cut][:, [1, 2]], fs[cut][:, [2, 0]]])
    ed.sort(1)
    ed, ct = np.unique(ed, axis=0, return_counts=True)
    boundary = ed[ct == 1]
    graph = {}
    for a, b in boundary:
        graph.setdefault(int(a), []).append(int(b))
        graph.setdefault(int(b), []).append(int(a))
    if not graph or any(len(v) != 2 for v in graph.values()):
        raise ValueError("Invalid external meatus entrance")
    start = next(iter(graph))
    order = [start]
    prev = -1
    current = start
    while True:
        following = next(j for j in graph[current] if j != prev)
        if following == start:
            break
        order.append(following)
        prev, current = current, following
        if len(order) > len(graph):
            raise ValueError("Broken meatus loop")
    directed = set(
        map(
            tuple,
            np.vstack(
                [fs[cut][:, [0, 1]], fs[cut][:, [1, 2]], fs[cut][:, [2, 0]]]
            ).tolist(),
        )
    )
    if (order[0], order[1]) not in directed:
        order = order[::-1]
    edgepoints = verts[order]
    deep = edgepoints - normal * 1.1
    end = deep.mean(0)
    offset = len(verts)
    verts = np.vstack([verts, deep, end])
    fs = fs[~cut]
    begin = len(fs)
    new = []
    for k, a in enumerate(order):
        j = (k + 1) % len(order)
        b = order[j]
        c = offset + k
        d = offset + j
        # Match the original front sheet orientation around the aperture.
        new.extend([(a, b, c), (b, d, c), (offset + len(order), c, d)])
    fs = np.vstack([fs, np.asarray(new, np.int32)])
    if side < 0:
        fs = fs[:, ::-1]
    return (
        verts,
        fs,
        np.flatnonzero(remove),
        {
            "root_vertices": n,
            "root_gap_mm": 0.0,
            "free_rim": True,
            "posterior_surface": True,
            "cartilage_free_lobule": True,
            "minimum_rear_sheet_standoff_mm": 1.15,
            "attachment_projection_mm": projection,
            "meatus_entrance_depth_mm": 1.1,
            "canal_faces_range": [begin, len(new)],
            "references": REFERENCES,
        },
    )
