"""A code-generated exterior chart over permitted internal anatomy.

No exterior mesh or image is read. Internal triangles define clearance constraints,
not a statistical face template. Atlas coordinates are retained without fitting.
The skin and each aperture annulus are newly generated in code.
"""

import hashlib
import heapq
import json
from pathlib import Path

import head_features as face
import numpy as np
import shapely
from ear_surface import smooth
from scipy.interpolate import CubicSpline, RectBivariateSpline

ROOT = Path(__file__).resolve().parent.parent


def load_internal_anatomy():
    geometry_path = ROOT / "assets/anatomy/geometry.npz"
    manifest_path = ROOT / "assets/anatomy/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = np.load(geometry_path)
    selected = []
    points = []
    for i, part in enumerate(manifest["parts"]):
        if part["kind"] not in ("bone", "muscle", "cartilage"):
            continue
        v = source[f"{i}_v"].astype(float) * 1000
        if v[:, 2].max() < -70 or np.abs(v[:, 0]).max() > 100 or v[:, 1].max() > 145:
            continue
        selected.append({"part": i, "name": part["name"], "kind": part["kind"]})
        points.append(v[(v[:, 2] >= -180.0001) & (v[:, 2] <= 147)])
    proof = {
        "source": "assets/anatomy/geometry.npz",
        "source_sha256": hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "frame": "workspace atlas millimetres; +X subject left / -Y anterior / Z superior",
        "transform": np.eye(4).tolist(),
        "parts": selected,
        "external_skin_inputs": [],
        "limitations": [
            "Chart containment is not global collision.",
            "One authored morphology, not a population model.",
        ],
    }
    return np.vstack(points), proof


class AnatomicalSurface:
    def __init__(self, contract):
        fields = face.FaceParameters.__dataclass_fields__
        self.p = face.FaceParameters(
            **{k: v for k, v in contract.items() if k in fields},
            skin_relief=0,
            eyelid_closure=0,
        )
        self.support_points, self.proof = load_internal_anatomy()
        self.zrange = (-180.0, 154.0)
        # Regional tissue depth is a single smoothly interpolated control lattice.
        self.depth = RectBivariateSpline(
            [-80, -65, -45, -25, 0, 25, 50, 80, 115, 148],
            [0, 15, 30, 45, 60, 80],
            [
                [7, 7, 7, 6, 4, 3],
                [6, 6, 7, 7, 5, 3],
                [5, 5, 7, 9, 6, 3],
                [4, 4, 7, 10, 6, 3],
                [2, 3, 6, 8, 5, 3],
                [2, 2, 3, 4, 4, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3],
            ],
            kx=3,
            ky=3,
            s=0,
        )
        self.loft_z = np.array(
            [
                -180,
                -150,
                -125,
                -100,
                -85,
                -75,
                -65,
                -45,
                -25,
                0,
                25,
                50,
                80,
                110,
                130,
                145,
                154.0,
            ]
        )
        self.loft = [
            CubicSpline(self.loft_z, v, bc_type="natural")
            for v in [
                [171, 95, 68, 56, 55, 55, 57, 61, 65, 72, 76, 82, 85, 78, 64, 43, 0],
                [
                    -55,
                    -45,
                    -35,
                    -30,
                    -49,
                    -65,
                    -79,
                    -80,
                    -77,
                    -77,
                    -76,
                    -76,
                    -77,
                    -66,
                    -45,
                    -13,
                    44,
                ],
                [
                    124,
                    119,
                    112,
                    103,
                    98,
                    95,
                    92,
                    91,
                    90,
                    105,
                    124,
                    132,
                    131,
                    121,
                    103,
                    73,
                    44,
                ],
            ]
        ]
        front_controls = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 1, 0, 0],
                [-2, -2, -2, -2, -1, 0, 0],
                [-4, -4, -3, -2, -1, 0, 0],
                [-6, -6, -4, -3, -3, 0, 0],
                [-19, -20, -12, -3, -3, 0, 0],
                [-29, -25, -10, -3, -4, 0, 0],
                [-24, -18, -8, -3, -4, 0, 0],
                [-17, -11, -5, -5, -5, 0, 0],
                [-11, -7, -3, -5, -4, 0, 0],
                [-5, -4, -3, -3, -2, 0, 0],
                [-2, -2, -2, -1, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        q = np.array([0, 0.12, 0.24, 0.42, 0.62, 0.82, 1])
        # Interpolate the full symmetric domain. Mirroring a one-sided spline
        # with abs(x) left a nonzero slope at the midline: a crease down the
        # bridge, tip, forehead and chin rather than a smooth bilateral face.
        self.front_chart = RectBivariateSpline(
            [-85, -70, -55, -40, -25, -15, -7, 5, 20, 30, 42, 55, 80, 110, 145, 154],
            np.r_[-q[:0:-1], q],
            np.c_[front_controls[:, :0:-1], front_controls],
            kx=3,
            ky=3,
            s=0,
        )
        self.proof["construction"] = (
            "authored section loft and one bicubic facial control lattice; explicit aperture annuli; independent internal-clearance check"
        )

    def eye(self, side):
        return np.array([side * 31.0, -61.0, 30.0])

    def mouth(self, x):
        q = np.clip(abs(x) / 25.5, 0, 1)
        span = np.maximum(1 - q * q, 0)
        line = -35.5 + 0.5 * q * q
        # The philtral peaks flank a lower central Cupid's-bow point.
        upper = (4.0 + 1.3 * np.exp(-((abs(x) - 7.0) / 4.0) ** 2)) * span**0.7
        return line, upper, 5.5 * span**0.7

    def eye_opening(self, x, side):
        q = (x - side * 31.0) * side / 13.25
        span = np.maximum(1 - q * q, 0)
        centre = 30.0 + 0.7 * q
        return q, centre + 4.3 * span**0.7, centre - 4.8 * span**0.75

    def tissue_depth(self, x, z):
        x, z = np.broadcast_arrays(x, z)
        return np.clip(
            self.depth.ev(
                np.clip(z, -80, 148).ravel(), np.clip(abs(x), 0, 80).ravel()
            ).reshape(x.shape),
            1.5,
            10.0,
        )

    def section(self, z):
        z = np.asarray(z)
        rx, front, back = (p(z) for p in self.loft)
        # The crown has a round pole, rather than the finite-slope tip of a
        # radius-versus-height interpolant. Blend its ellipse into the vault.
        crown = np.sqrt(np.maximum(1 - ((z - 75.0) / 79.0) ** 2, 0))
        t = smooth(80, 110, z)
        centre = (front + back) / 2
        rx = rx * (1 - t) + 85 * crown * t
        half_depth = (back - front) / 2 * (1 - t) + 105 * crown * t
        return rx, centre - half_depth, centre + half_depth

    def back(self, x, z):
        rx, front, back = self.section(z)
        q = np.clip(abs(x) / np.maximum(rx, 0.001), 0, 1)
        power = 2.0 + 1.0 - smooth(-65, -35, np.asarray(z))
        return (front + back) / 2 + (back - front) / 2 * np.maximum(
            1 - q**power, 0
        ) ** (1 / power)

    eye_support = face.eye_support

    def uv(self, vertices):
        x, y, z = np.asarray(vertices).T
        rx, front, back = self.section(z)
        theta = np.arcsin(np.clip(x / np.maximum(rx, 0.001), -1, 1))
        theta = np.where(
            y > (front + back) / 2,
            np.where(x >= 0, np.pi - theta, -np.pi - theta),
            theta,
        )
        return np.column_stack(
            [
                0.5 + theta / (2 * np.pi),
                (z - self.zrange[0]) / (self.zrange[1] - self.zrange[0]),
            ]
        )

    def base_front(self, x, z):
        x, z = np.broadcast_arrays(x, z)
        rx, front, back = self.section(z)
        q = np.clip(abs(x) / np.maximum(rx, 0.001), 0, 1)
        c = np.sqrt(np.maximum(1 - q * q, 0))
        y = (front + back) / 2 - (back - front) / 2 * c * (1 + 0.18 * q * q)
        offset = self.front_chart.ev(np.clip(z, -85, 154).ravel(), q.ravel()).reshape(
            x.shape
        )
        return y + offset * c

    def lid_skin(self, x, z, base):
        result = np.array(base, copy=True)
        for side in [-1, 1]:
            c = self.eye(side)
            q, upper, lower = self.eye_opening(x, side)
            distance = np.where(z > 30, z - upper, lower - z)
            weight = (1 - smooth(0, 7, np.maximum(distance, 0))) * (
                1 - smooth(0.88, 1.3, abs(q))
            )
            # Use the same globe/corner support as the conjunctival sleeve.
            # A clamped sphere created a flat shelf outside its equator.
            support = self.eye_support(x, z, side, base=base) - 0.35
            # A continuous tarsal roll connects the fissure to the orbital chart.
            result = result * (1 - weight) + support * weight
            inside_globe = (x - c[0]) ** 2 + (z - c[2]) ** 2 < face.EYE_RADIUS**2
            outside_fissure = (z >= upper) | (z <= lower) | (abs(q) >= 1)
            result = np.where(
                inside_globe & outside_fissure,
                face.smooth_minimum(result, support, 0.4),
                result,
            )
            crease = 0.35 * np.exp(-((z - upper - 4.5) / 0.8) ** 2)
            result += crease * (1 - smooth(0.7, 1.1, abs(q)))
        line, upper, lower = self.mouth(x)
        d = z - line
        h = np.where(d >= 0, upper, lower)
        t = np.clip(abs(d) / np.maximum(h, 1e-6), 0, 1)
        span = np.maximum(1 - (x / 25.5) ** 2, 0)
        # Both lips meet at the same depth, with their convex body away from
        # the closure line rather than a step between two ridge maxima.
        lip = (1.4 * (1 - t) ** 2 + np.where(d >= 0, 2.2, 2.9) * np.sin(np.pi * t) ** 1.5) * span**0.7
        result -= lip
        return result

    def front(self, x, z):
        return self.lid_skin(x, z, self.base_front(x, z))

    def point(self, theta, z):
        theta, z = np.broadcast_arrays(theta, z)
        rx, _, _ = self.section(z)
        x = rx * np.sin(theta)
        fy = self.front(x, z)
        y = np.where(np.cos(theta) >= 0, fy, self.back(x, z))
        return np.stack([x, y, z], axis=-1)


def refine_patch(patch, vertices, anatomy):
    """Bisect interior edges conformingly; preserve both aperture boundaries."""
    coords = list(vertices)
    triangles = {i: tuple(map(int, t)) for i, t in enumerate(patch)}
    adjacency = {}
    queue = []
    serial = len(triangles)

    def add(tid, tri):
        triangles[tid] = tri
        for j in range(3):
            edge = tuple(sorted((tri[j], tri[(j + 1) % 3])))
            adjacency.setdefault(edge, set()).add(tid)
            length = np.linalg.norm(
                np.asarray(coords[edge[0]])[[0, 2]]
                - np.asarray(coords[edge[1]])[[0, 2]]
            )
            heapq.heappush(queue, (-float(length), edge))

    for tid, tri in list(triangles.items()):
        add(tid, tri)
    boundary = {edge for edge, owners in adjacency.items() if len(owners) == 1}
    while queue:
        negative, edge = heapq.heappop(queue)
        if -negative <= 1.25:
            break
        if edge in boundary or len(adjacency.get(edge, ())) != 2:
            continue
        owners = list(adjacency[edge])
        mid = (np.asarray(coords[edge[0]]) + np.asarray(coords[edge[1]])) / 2
        mid[1] = float(anatomy.front(np.asarray(mid[0]), np.asarray(mid[2])))
        middle = len(coords)
        coords.append(mid)
        for tid in owners:
            tri = triangles.pop(tid)
            for j in range(3):
                e = tuple(sorted((tri[j], tri[(j + 1) % 3])))
                adjacency[e].discard(tid)
            j = next(
                j for j in range(3) if tuple(sorted((tri[j], tri[(j + 1) % 3]))) == edge
            )
            u, v, w = tri[j], tri[(j + 1) % 3], tri[(j + 2) % 3]
            add(serial, (u, middle, w))
            serial += 1
            add(serial, (middle, v, w))
            serial += 1
        if len(coords) - len(vertices) > 30000:
            raise ValueError("Aperture refinement exceeded its construction budget")
    result = np.asarray(list(triangles.values()))
    p = np.asarray(coords)[result][:, :, [0, 2]]
    sign = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 1, 1] - p[:, 0, 1]
    ) * (p[:, 2, 0] - p[:, 0, 0])
    result[sign < 0] = result[sign < 0][:, ::-1]
    return result, np.asarray(coords)


def head_mesh(a, rows=370, cols=769):
    """Structured skull chart with explicit annular aperture topology."""
    theta = np.linspace(-np.pi, np.pi, cols)
    zz = np.linspace(*a.zrange, rows)
    tt, z = np.meshgrid(theta, zz)
    vertices = a.point(tt, z).reshape(-1, 3)
    faces = face.grid_faces(rows, cols)
    holes = [
        ("eye", -31, 30, 20, 12, -1),
        ("eye", 31, 30, 20, 12, 1),
        ("mouth", 0, -35.5, 29, 9, 0),
        ("nostril", -9.35, -17.8, 4.4, 2.4, -1),
        ("nostril", 10.05, -17.8, 4.4, 2.4, 1),
    ]
    proof = []
    for kind, cx, cz, rx, rz, side in holes:
        centre = vertices[faces].mean(1)
        remove = ((centre[:, 0] - cx) / rx) ** 2 + ((centre[:, 2] - cz) / rz) ** 2 < 1
        remove &= centre[:, 1] < 0
        removed = faces[remove]
        faces = faces[~remove]
        edges = np.sort(
            np.vstack([removed[:, [0, 1]], removed[:, [1, 2]], removed[:, [2, 0]]]),
            axis=1,
        )
        edges, count = np.unique(edges, axis=0, return_counts=True)
        edges = edges[count == 1]
        adj = {}
        for u, v in edges:
            adj.setdefault(int(u), []).append(int(v))
            adj.setdefault(int(v), []).append(int(u))
        if not adj or any(len(n) != 2 for n in adj.values()):
            raise ValueError(
                "Aperture boundary must be one closed degree-two loop: " + kind
            )
        loop = [next(iter(adj))]
        prev = -1
        while True:
            nxt = next(n for n in adj[loop[-1]] if n != prev)
            if nxt == loop[0]:
                break
            prev = loop[-1]
            loop.append(nxt)
            if len(loop) > len(adj):
                raise ValueError("Aperture loop did not close")
        if len(loop) != len(adj):
            raise ValueError("Disconnected aperture boundary")
        outer = vertices[loop].copy()
        raw = np.unwrap(np.arctan2((outer[:, 2] - cz) / rz, (outer[:, 0] - cx) / rx))
        direction = 1 if raw[-1] > raw[0] else -1
        count = 96 if kind == "eye" else 128 if kind == "mouth" else 64
        angle = raw[0] + direction * np.linspace(0, 2 * np.pi, count, endpoint=False)
        if kind == "eye":
            ix = cx + 13.25 * np.cos(angle)
            _, up, low = a.eye_opening(ix, side)
            iz = np.where(np.sin(angle) >= 0, up, low)
        elif kind == "mouth":
            ix = 25.5 * np.cos(angle)
            line, _, _ = a.mouth(ix)
            iz = (
                line + 0.16 * np.sin(angle) * np.maximum(1 - (ix / 25.5) ** 2, 0) ** 0.3
            )
        else:
            ix = cx + 3.15 * np.cos(angle)
            iz = cz + 1.35 * np.sin(angle)
        inner = np.column_stack([ix, a.front(ix, iz), iz])
        inner_ids = np.arange(len(vertices), len(vertices) + len(inner))
        vertices = np.vstack([vertices, inner])
        polygon = shapely.Polygon(outer[:, [0, 2]], holes=[inner[:, [0, 2]]])
        if not polygon.is_valid:
            raise ValueError("Invalid aperture polygon")
        lookup = {
            tuple(np.round(v, 9)): int(i)
            for v, i in zip(
                np.vstack([outer[:, [0, 2]], inner[:, [0, 2]]]), np.r_[loop, inner_ids]
            )
        }
        triangles = shapely.constrained_delaunay_triangles(polygon)
        patch = []
        for tri in triangles.geoms:
            patch.append(
                [
                    lookup[tuple(np.round(p, 9))]
                    for p in np.asarray(tri.exterior.coords)[:3]
                ]
            )
        patch = np.asarray(patch)
        patch, vertices = refine_patch(patch, vertices, a)
        projected = vertices[patch][:, :, [0, 2]]
        signed = (projected[:, 1, 0] - projected[:, 0, 0]) * (
            projected[:, 2, 1] - projected[:, 0, 1]
        ) - (projected[:, 1, 1] - projected[:, 0, 1]) * (
            projected[:, 2, 0] - projected[:, 0, 0]
        )
        if (signed <= 0).any():
            raise ValueError("Folded aperture triangulation")
        faces = np.vstack([faces, patch])
        proof.append(
            {
                "feature": kind,
                "side": side,
                "loop_vertices": len(loop),
                "triangles": len(patch),
                "minimum_projected_area_mm2": float(signed.min() / 2),
                "triangulation": "constrained Delaunay; conforming interior-edge refinement; shared boundary",
            }
        )
    return face.surface_part("Continuous_anatomical_skin", vertices, faces), proof
