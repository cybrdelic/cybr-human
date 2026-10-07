"""Code-authored feature utilities; the head chart lives in anatomical_surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfacePart:
    """Authored mesh arrays and export metadata; independent of gallery tools."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    material: int = 0
    group: str = "anatomy"
    provenance: str = "original-procedural"
    role: str = ""
    tags: tuple = ()
    portrait_uv: np.ndarray | None = None


@dataclass(frozen=True)
class FaceParameters:
    seed: int = 271828
    eye_spacing: float = 61.0
    eye_height: float = 30.0
    nose_projection: float = 18.5
    mouth_width: float = 51.0
    jaw_width: float = 1.0
    skull_width: float = 0.96
    brow_weight: float = 1.0
    skin_relief: float = 0.0025
    eyelid_closure: float = 0.06
    eye_width: float = 28.0
    eye_opening: float = 9.5
    eye_tilt: float = 1.1
    cheek_width: float = 1.0
    chin_width: float = 0.94
    nose_width: float = 1.0
    upper_lip_fullness: float = 0.78
    lower_lip_fullness: float = 0.88
    skin_tone: float = 1.0
    asymmetry: float = 0.6
    scalp_hair: bool = False


ZMIN, ZMAX = -180.0, 129.0

SKIN, LID, CAVITY, SCLERA, IRIS, PUPIL, CORNEA, HAIR = range(8)

EYE_RADIUS = 12.1


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def smooth_minimum(a, b, width):
    h = np.maximum(width - np.abs(a - b), 0) / width
    return np.minimum(a, b) - h * h * width * 0.25


def eye_support(self, x, z, side, base=None):
    """Globe and conjunctival corner support, in millimetres.

    The 28 mm palpebral fissure extends beyond the 24.2 mm globe. Its
    corners therefore rest on a moist conjunctival sleeve rather than an
    artificially widened eyeball. A tangent-continuous continuation of
    the globe provides that separately authored sleeve's surface.
    """
    c = self.eye(side)
    lx = (x - c[0]) * side
    r = np.hypot(lx, z - c[2])
    join = 11.0
    depth = np.sqrt(EYE_RADIUS**2 - join**2)
    globe = np.sqrt(np.maximum(EYE_RADIUS**2 - r * r, 0))
    sleeve = depth - (join / depth) * 1.2 * (1 - np.exp(-np.maximum(r - join, 0) / 1.2))
    support = np.where(r <= join, globe, sleeve)
    cornea = 1.28 * np.exp(-(((r * r) / (5.5**2)) ** 2))
    surface = c[1] - support - cornea
    # The medial canthus joins the bridge's anterior tissue, rather than
    # being recessed to the side of a free-standing sphere. Keeping that
    # actual orbital attachment avoids a deep, artificial corner cavity.
    medial = smoothstep(9.6, self.p.eye_width / 2, -lx)
    attachment = self.base_front(x, z) if base is None else base
    return surface * (1 - medial) + (attachment + 0.14) * medial


def surface_part(name, vertices, faces, material=SKIN, uv=None, role=""):
    v = np.ascontiguousarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.ascontiguousarray(faces, dtype=np.int32).reshape(-1, 3)
    area = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    keep = np.linalg.norm(area, axis=1) > 1e-10
    f = f[keep]
    area = area[keep]
    n = np.zeros_like(v)
    for i in range(3):
        np.add.at(n, f[:, i], area)
    n = unit(n)
    p = SurfacePart(
        name,
        v,
        f,
        n,
        material=material,
        group="anatomy",
        provenance="original-procedural",
        role=role or name,
        tags=("original-procedural", "no-scan", "deterministic"),
    )
    p.portrait_uv = (
        np.zeros((len(v), 2)) if uv is None else np.asarray(uv, float).reshape(-1, 2)
    )
    return p


def grid_faces(rows, cols, reverse=False):
    a = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    f = np.concatenate(
        [
            np.stack([a, a + 1, a + cols], 1),
            np.stack([a + 1, a + cols + 1, a + cols], 1),
        ]
    )
    return f[:, ::-1] if reverse else f


def ellipsoid(name, center, radii, material, nu=128, nv=80, iris_cut=False):
    if iris_cut:
        # A polar chart about the ocular axis has an exact circular limbus.
        # Removing triangles by their centroids left a jagged, overlapping
        # sclera/iris seam which was visible even in the undeformed head.
        angle = np.linspace(0, 2 * np.pi, nu + 1)
        phi = np.linspace(np.arcsin(5.72 / radii[0]), np.pi - 0.0001, nv + 1)
        aa, pp = np.meshgrid(angle, phi)
        q = np.stack(
            [np.sin(pp) * np.cos(aa), -np.cos(pp), np.sin(pp) * np.sin(aa)], -1
        )
        v = (q * np.asarray(radii) + center).reshape(-1, 3)
        return surface_part(name, v, grid_faces(nv + 1, nu + 1, reverse=True), material)
    theta = np.linspace(-np.pi, np.pi, nu + 1)
    phi = np.linspace(0.0001, np.pi - 0.0001, nv + 1)
    tt, pp = np.meshgrid(theta, phi)
    q = np.stack([np.sin(pp) * np.sin(tt), -np.sin(pp) * np.cos(tt), np.cos(pp)], -1)
    v = (q * np.array(radii) + center).reshape(-1, 3)
    f = grid_faces(nv + 1, nu + 1, reverse=True)
    uv = np.column_stack(
        [
            0.5 + (v[:, 0] - center[0]) / (2 * radii[0]),
            0.5 + (v[:, 2] - center[2]) / (2 * radii[2]),
        ]
    )
    return surface_part(name, v, f, material, uv)


def disk(name, center, radius, material, yfunc, rmin=0.0, nr=32, nt=192):
    r = np.linspace(rmin, radius, nr)
    theta = np.linspace(0, 2 * np.pi, nt + 1)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    x = rr * np.cos(tt)
    z = rr * np.sin(tt)
    y = yfunc(rr)
    v = np.stack([x + center[0], y + center[1], z + center[2]], -1).reshape(-1, 3)
    uv = np.column_stack(
        [0.5 + x.ravel() / (2 * radius), 0.5 + z.ravel() / (2 * radius)]
    )
    return surface_part(name, v, grid_faces(nr, nt + 1, reverse=True), material, uv)


def tube_collection(name, curves, radii, material, a=None, sides=5):
    vs = []
    fs = []
    offset = 0
    for curve, radius in zip(curves, radii):
        c = np.asarray(curve, float)
        tangent = unit(np.gradient(c, axis=0))
        ref = np.tile([0.0, 1.0, 0.0], (len(c), 1))
        b = unit(np.cross(tangent, ref))
        n = unit(np.cross(b, tangent))
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        radius = np.broadcast_to(radius, (len(c),))
        v = c[:, None, :] + radius[:, None, None] * (
            np.cos(angles)[None, :, None] * b[:, None, :]
            + np.sin(angles)[None, :, None] * n[:, None, :]
        )
        i = np.arange((len(c) - 1) * sides).reshape(-1, sides)
        j = np.roll(i, -1, axis=1)
        f = np.concatenate(
            [
                np.stack([i, j, i + sides], -1).reshape(-1, 3),
                np.stack([j, j + sides, i + sides], -1).reshape(-1, 3),
            ]
        )
        # Close tube tips; even fine lashes have real, finite geometry.
        if sides > 2:
            cap0 = np.array([[0, k + 1, k] for k in range(1, sides - 1)])
            cap1 = np.array(
                [
                    [
                        (len(c) - 1) * sides,
                        (len(c) - 1) * sides + k,
                        (len(c) - 1) * sides + k + 1,
                    ]
                    for k in range(1, sides - 1)
                ]
            )
            f = np.concatenate([f, cap0, cap1])
        vs.append(v.reshape(-1, 3))
        fs.append(f + offset)
        offset += v.size // 3
    v = np.concatenate(vs)
    return surface_part(name, v, np.concatenate(fs), material, a.uv(v) if a else None)


def conjunctival_crescents(a, side):
    """Actual corner tissue between the spherical eye and its wider lid slit.

    The inner edge lies just beneath the sclera. Both outer edges extend
    underneath the analytic lid margin, so the corner is a supported surface
    rather than a black void or an enlarged patch of white eyeball.
    """
    if a.p.eyelid_closure >= 1:
        return []
    c = a.eye(side)
    parts = []
    for direction, label in [(-1, "medial"), (1, "lateral")]:
        horizontal = np.linspace(10.2, a.p.eye_width / 2, 97)[:, None]
        x = np.broadcast_to(c[0] + side * direction * horizontal, (97, 33))
        _, upper, lower = a.eye_opening(x, side)
        t = np.linspace(0, 1, 33)[None, :]
        z = lower + (upper - lower) * t
        offset = 0.03 - 0.07 * smoothstep(10.2, 11.7, horizontal)
        y = a.eye_support(x, z, side) + offset
        border = 1 - smoothstep(0, 0.16, np.minimum(t, 1 - t))
        y = y * (1 - border) + (a.front(x, z) + 0.03) * border
        vertices = np.stack([x, y, z], -1).reshape(-1, 3)
        faces = grid_faces(97, 33, reverse=side * direction > 0)
        part = surface_part(
            ("Right" if side < 0 else "Left") + "_" + label + "_conjunctival_sleeve",
            vertices,
            faces,
            LID,
            a.uv(vertices),
            role="Authored moist conjunctival corner tissue supporting the lid",
        )
        # All visible surfaces face anteriorly regardless of eye or corner.
        if np.mean(part.normals[:, 1]) > 0:
            part = surface_part(
                part.name, vertices, part.faces[:, ::-1], LID, a.uv(vertices), part.role
            )
        parts.append(part)
    return parts


def nasal_cavity(a, side):
    cx = side * 9.7 + 0.35
    cz = -17.8
    theta = np.linspace(0, 2 * np.pi, 129)
    r = np.linspace(0, 1, 24)[:, None]
    x = cx + 3.26 * r * np.cos(theta)
    z = cz + 1.48 * r * np.sin(theta)
    y = a.front(x, z) + 6.5 * (1 - r * r)
    v = np.stack([x, y, z], -1).reshape(-1, 3)
    prefix = ("Right" if side < 0 else "Left") + "_nasal_vestibule"
    full = surface_part(prefix, v, grid_faces(24, 129, reverse=True), SKIN, a.uv(v))
    mid = full.vertices[full.faces].mean(axis=1)
    radial = ((mid[:, 0] - cx) / 3.26) ** 2 + ((mid[:, 2] - cz) / 1.48) ** 2
    parts = []
    for name, mask, material in [
        ("rim", radial >= 0.65, SKIN),
        ("interior", radial < 0.65, CAVITY),
    ]:
        faces = full.faces[mask]
        indices, inverse = np.unique(faces, return_inverse=True)
        part = SurfacePart(
            prefix + "_" + name,
            full.vertices[indices],
            inverse.reshape(-1, 3),
            full.normals[indices],
            material=material,
            group="anatomy",
            provenance="original-procedural",
            role="Continuous nasal tissue rim and recessed vestibule",
        )
        part.portrait_uv = full.portrait_uv[indices]
        parts.append(part)
    return parts


def mouth_interior(a):
    x = np.linspace(-25, 25, 180)
    w = np.linspace(-1, 1, 12)
    xx, ww = np.meshgrid(x, w)
    line, _, _ = a.mouth(xx)
    z = line + ww * (1.1 * np.maximum(1 - (xx / 25) ** 2, 0) ** 0.6 + 0.04)
    y = a.front(xx, z) + 3.4 + 1.5 * (1 - ww * ww)
    v = np.stack([xx, y, z], -1).reshape(-1, 3)
    return surface_part(
        "Recessed_closed_mouth_cavity", v, grid_faces(12, 180), CAVITY, a.uv(v)
    )


def eyebrow_and_lashes(a, side, margin, rng):
    curves = []
    radii = []
    for _ in range(900):
        q = rng.beta(1.35, 1.3)
        x = side * (15.4 + 39 * q)
        z = (
            44.0
            + 4.8 * np.sin(np.pi * q * 0.91)
            - 2.7 * q
            + rng.normal(0, 1.3) * (1 - 0.5 * q)
        )
        length = rng.uniform(3.4, 6.0) * (1 - 0.26 * q)
        dx = side * (0.35 + 0.62 * q) * length
        dz = (0.96 - 0.9 * q) * length
        t = np.linspace(0, 1, 6)
        xx = x + dx * t
        zz = z + dz * t
        yy = a.front(xx, zz) - 0.08 - 0.24 * np.sin(np.pi * t)
        curves.append(np.column_stack([xx, yy, zz]))
        radii.append(rng.uniform(0.03, 0.055) * (1 - 0.85 * t))
    for upper, count in [(True, 75), (False, 36)]:
        angles = (
            np.linspace(0.08, np.pi - 0.08, count)
            if upper
            else np.linspace(np.pi + 0.12, 2 * np.pi - 0.12, count)
        )
        for angle in angles:
            j = int(angle / (2 * np.pi) * (len(margin) - 1))
            root = margin[j].copy()
            root[1] -= 0.09
            t = np.linspace(0, 1, 6)
            length = rng.uniform(4.0, 6.2) if upper else rng.uniform(1.7, 3.0)
            xx = root[0] + side * np.cos(angle) * 1.5 * t
            yy = root[1] - length * (0.73 * t - 0.2 * t * t)
            zz = root[2] + (1 if upper else -1) * length * 0.57 * t * t
            curves.append(np.column_stack([xx, yy, zz]))
            radii.append((0.032 if upper else 0.02) * (1 - 0.94 * t))
    return tube_collection(
        ("Right" if side < 0 else "Left") + "_individual_brow_hairs_and_lashes",
        curves,
        radii,
        HAIR,
        a,
        5,
    )
