"""Bounded floating-point rest topology, boundary intersection and globe audit.

Run under cybr-heavy.lock, with OPENBLAS_NUM_THREADS=1 and an external 120s
timeout. Does not alter candidate geometry or make mechanics/contact claims.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mandibular_collision_audit import broad_pairs, intersect_sat

TOL = 1e-9


def guard(start):
    if time.monotonic() - start > 110:
        raise TimeoutError("110-second internal guard reached")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 300 * 1024:
        raise MemoryError("300 MiB RSS guard reached")


def clip_coplanar(a, b, normal):
    """Intersection polygon, projected into the best-conditioned plane."""
    axes = np.delete(np.arange(3), np.argmax(abs(normal)))
    a2, b2 = a[:, axes], b[:, axes]
    cross = lambda p, q: p[0] * q[1] - p[1] * q[0]
    sign = np.sign(cross(b2[1] - b2[0], b2[2] - b2[0]))
    poly = [p.copy() for p in a]
    for k in range(3):
        origin, end = b2[k], b2[(k + 1) % 3]
        edge = end - origin
        old, poly = poly, []
        if not old:
            break
        for first, second in zip(old, old[1:] + old[:1]):
            d1 = sign * cross(edge, first[axes] - origin)
            d2 = sign * cross(edge, second[axes] - origin)
            if d1 >= 0:
                poly.append(first)
            if (d1 >= 0) != (d2 >= 0):
                poly.append(first + d1 / (d1 - d2) * (second - first))
    return np.asarray(poly).reshape(-1, 3)


def plane_section(triangle, origin, normal):
    distances = (triangle - origin) @ normal
    points = [triangle[i] for i in range(3) if abs(distances[i]) <= TOL]
    for i in range(3):
        j = (i + 1) % 3
        if distances[i] * distances[j] < 0:
            points.append(triangle[i] + distances[i] / (distances[i] - distances[j])
                          * (triangle[j] - triangle[i]))
    return np.asarray(points).reshape(-1, 3)


def shared_intersection(a, b, common):
    """Classify SAT hits; shared features are checked, never discarded wholesale."""
    na = np.cross(a[1] - a[0], a[2] - a[0])
    nb = np.cross(b[1] - b[0], b[2] - b[0])
    if min(np.linalg.norm(na), np.linalg.norm(nb)) <= 1e-13:
        return "unresolved_degenerate", []
    na /= np.linalg.norm(na)
    nb /= np.linalg.norm(nb)
    line = np.cross(na, nb)
    line_length = np.linalg.norm(line)
    if line_length < 1e-10:
        if max(abs((b - a[0]) @ na)) > TOL:
            return "unresolved_near_parallel", []
        points = clip_coplanar(a, b, na)
    else:
        line /= line_length
        pa, pb = plane_section(a, b[0], nb), plane_section(b, a[0], na)
        if not len(pa) or not len(pb):
            return "unresolved_section", []
        origin = common[0]
        ia, ib = (pa - origin) @ line, (pb - origin) @ line
        lo, hi = max(ia.min(), ib.min()), min(ia.max(), ib.max())
        if lo > hi + TOL:
            return "unresolved_separation", []
        points = origin + np.asarray([lo, hi])[:, None] * line
    if not len(points):
        return "unresolved_empty_clip", []
    if len(common) == 1:
        distances = np.linalg.norm(points - common[0], axis=1)
    elif len(common) == 2:
        edge = common[1] - common[0]
        alpha = np.clip((points - common[0]) @ edge / (edge @ edge), 0, 1)
        distances = np.linalg.norm(points - common[0] - alpha[:, None] * edge, axis=1)
    else:
        return "duplicate_boundary_triangle", points[:4].tolist()
    if distances.max() > TOL:
        return "beyond_shared_feature", points[np.argsort(-distances)[:4]].tolist()
    return "expected_shared_feature", []


def closest_triangles(center, triangle):
    ab, ac = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    normal = np.cross(ab, ac)
    normal2 = (normal * normal).sum(1)
    dp = ((center - triangle[:, 0]) * normal).sum(1)
    projection = center - dp[:, None] / normal2[:, None] * normal
    rr = projection - triangle[:, 0]
    aa, bb, cc = (ab * ab).sum(1), (ac * ac).sum(1), (ab * ac).sum(1)
    ar, br = (ab * rr).sum(1), (ac * rr).sum(1)
    denominator = aa * bb - cc * cc
    u, v = (ar * bb - br * cc) / denominator, (br * aa - ar * cc) / denominator
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    distance2 = np.where(inside, dp * dp / normal2, np.inf)
    closest = projection.copy()
    for k in range(3):
        edge = triangle[:, (k + 1) % 3] - triangle[:, k]
        alpha = np.clip(((center - triangle[:, k]) * edge).sum(1)
                        / (edge * edge).sum(1), 0, 1)
        point = triangle[:, k] + alpha[:, None] * edge
        dd = ((point - center) ** 2).sum(1)
        take = dd < distance2
        distance2[take], closest[take] = dd[take], point[take]
    return np.sqrt(distance2), closest


def fixtures():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    cases = [
        (np.array([[0, 0, 0], [1, 0, 0], [0, -1, 0]], float), a[:2], "expected_shared_feature"),
        (np.array([[0, 0, 0], [1, 0, 0], [.2, .4, 0]], float), a[:2], "beyond_shared_feature"),
        (np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], float), a[:1], "expected_shared_feature"),
        (np.array([[0, 0, 0], [.5, .1, 0], [.1, .5, 0]], float), a[:1], "beyond_shared_feature"),
        (np.array([[0, 0, 0], [.6, .2, 1], [.6, .2, -1]], float), a[:1], "beyond_shared_feature"),
        (np.array([[0, 0, 0], [-.6, -.2, 1], [-.6, -.2, -1]], float), a[:1], "expected_shared_feature"),
        (np.array([[0, 0, 0], [1, 0, 0], [.5, 0, 1]], float), a[:2], "expected_shared_feature"),
    ]
    for b, common, expected in cases:
        assert intersect_sat(a, b[None], tol=TOL)[0]
        assert shared_intersection(a, b, common)[0] == expected


def bad_boundary_vertex_links(boundary):
    """Every boundary vertex link must be one cycle, not several touching fans."""
    links = defaultdict(list)
    for a, b, c in boundary.tolist():
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    bad = []
    for vertex, edges in links.items():
        graph = defaultdict(set)
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        first = next(iter(graph))
        seen, todo = set(), [first]
        while todo:
            node = todo.pop()
            if node not in seen:
                seen.add(node)
                todo.extend(graph[node] - seen)
        if len(seen) != len(graph) or any(len(adj) != 2 for adj in graph.values()):
            bad.append(int(vertex))
    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    fixtures()
    report = {
        "schema": "connected-midface-rest-boundary-audit-v1",
        "source_geometry": str(args.geometry),
        "source_sha256": hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        "status": "running",
        "fixtures_passed": True,
        "separation_tolerance_design_units": TOL,
        "limits": [
            "Float64 AABB/SAT and geometric intersection tests; not exact-arithmetic certification.",
            "Shared vertex/edge intersections are classified against the common feature, not skipped.",
            "Boundary self-intersection check does not directly test every pair of interior tetrahedra.",
            "Globe check uses the supplied native analytic spheres; no other anatomy or support is tested.",
            "No bone, contact, mechanical, physiological, or dynamic acceptance is implied.",
        ],
    }
    try:
        data = np.load(args.geometry)
        x, t = data["vertices"], data["tetrahedra"]
        dm = x[t[:, 1:]] - x[t[:, 0, None]]
        determinant = np.einsum("ij,ij->i", dm[:, 0], np.cross(dm[:, 1], dm[:, 2]))
        faces = np.concatenate([t[:, [j for j in range(4) if j != k]] for k in range(4)])
        for k in (1, 3):
            block = faces[k * len(t):(k + 1) * len(t)]
            block[:, [1, 2]] = block[:, [2, 1]]
        canonical = np.sort(faces, axis=1)
        unique, first, inverse, count = np.unique(canonical, axis=0, return_index=True,
                                                  return_inverse=True, return_counts=True)
        parity = np.prod(np.sign(faces[:, [1, 2, 2]] - faces[:, [0, 0, 1]]), axis=1)
        orientation = np.bincount(inverse, weights=parity)
        boundary = faces[first[count == 1]]
        owner_tets = np.tile(np.arange(len(t)), 4)[first[count == 1]]
        directed = np.concatenate([boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]]])
        _, einverse, ecount = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        eorientation = np.bincount(einverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
        parents = np.arange(len(t))
        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        order = np.argsort(inverse, kind="stable")
        offsets = np.r_[0, np.cumsum(count)[:-1]]
        double = np.flatnonzero(count == 2)
        owners = np.tile(np.arange(len(t)), 4)
        for aa, bb in zip(owners[order[offsets[double]]], owners[order[offsets[double] + 1]]):
            aa, bb = root(aa), root(bb)
            if aa != bb:
                parents[bb] = aa
        topology = dict(nodes=len(x), tetrahedra=len(t), boundary_triangles=len(boundary),
                        minimum_determinant=float(determinant.min()),
                        nonpositive_tetrahedra=int((determinant <= 0).sum()),
                        duplicate_tetrahedra=len(t) - len(np.unique(np.sort(t, axis=1), axis=0)),
                        nonmanifold_tet_faces=int((count > 2).sum()),
                        inconsistent_internal_face_winding=int(((count == 2) & (orientation != 0)).sum()),
                        shared_face_components=len({root(i) for i in range(len(t))}),
                        nonmanifold_boundary_edges=int((ecount != 2).sum()),
                        inconsistent_boundary_edge_winding=int((eorientation != 0).sum()),
                        nonmanifold_boundary_vertex_links=bad_boundary_vertex_links(boundary))
        report["topology"] = topology
        print("TOPOLOGY " + json.dumps(topology), flush=True)
        triangle = x[boundary]
        report["globes"] = []
        cross23 = np.cross(dm[:, 1], dm[:, 2])
        cross31 = np.cross(dm[:, 2], dm[:, 0])
        cross12 = np.cross(dm[:, 0], dm[:, 1])
        for side, center in [("right", [31.3, 12, 94.6]), ("left", [-31.3, 12, 94.6])]:
            center = np.asarray(center)
            distance, closest = closest_triangles(center, triangle)
            j = int(distance.argmin())
            rr = center - x[t[:, 0]]
            weights = np.stack([np.einsum("ij,ij->i", rr, g) / determinant
                                for g in (cross23, cross31, cross12)], axis=1)
            contains = (weights.min(1) >= 0) & (weights.sum(1) <= 1)
            globe = dict(side=side, center=center.tolist(), radius=13.161,
                         boundary_clearance=float(distance[j] - 13.161),
                         center_inside_tetrahedra=int(contains.sum()),
                         boundary_triangles_inside_sphere=int((distance < 13.161).sum()),
                         closest_boundary_triangle=j, closest_point=closest[j].tolist(),
                         closest_owner_tet=int(owner_tets[j]))
            report["globes"].append(globe)
            print("GLOBE " + json.dumps(globe), flush=True)
        guard(start)
        candidates, hits, classes = Counter(), Counter(), Counter()
        examples = []
        checked = 0
        report["boundary_self_intersections"] = dict(examples=examples)
        for i, ids in broad_pairs(triangle, triangle, cell=4.):
            ids = ids[ids > i]
            if not len(ids):
                continue
            common_count = (boundary[ids, :, None] == boundary[i][None, None, :]).sum((1, 2))
            candidates.update(map(int, common_count))
            hit_mask = intersect_sat(triangle[i], triangle[ids], tol=TOL)
            for j, shared in zip(ids[hit_mask], common_count[hit_mask]):
                hits[int(shared)] += 1
                common_ids = np.intersect1d(boundary[i], boundary[j])
                if not shared:
                    classification, witness = "nonadjacent_intersection_or_touch", []
                else:
                    classification, witness = shared_intersection(triangle[i], triangle[j], x[common_ids])
                classes[classification] += 1
                if classification != "expected_shared_feature" and len(examples) < 40:
                    item = dict(boundary_triangles=[int(i), int(j)], shared_nodes=common_ids.tolist(),
                                classification=classification, witness_points=witness,
                                owner_tetrahedra=[int(owner_tets[i]), int(owner_tets[j])],
                                cell_parameters=data["cell_parameters"][owner_tets[[i, j]]].tolist())
                    examples.append(item)
                    if len(examples) <= 5:
                        print("BOUNDARY_VIOLATION " + json.dumps(item), flush=True)
            checked = i + 1
            if i % 64 == 0:
                guard(start)
            report["boundary_self_intersections"].update(
                checked_triangles=checked, total_triangles=len(triangle),
                candidate_pairs_by_shared_nodes=dict(candidates),
                sat_hits_by_shared_nodes=dict(hits), classification_counts=dict(classes))
        report["boundary_self_intersections"]["checked_triangles"] = len(triangle)
        report["boundary_self_intersections"]["complete"] = True
        report["status"] = "completed_floating_point_audit"
    except (TimeoutError, MemoryError) as exc:
        report["status"] = "incomplete"
        report["blocker"] = str(exc)
    finally:
        report["seconds"] = time.monotonic() - start
        report["maximum_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("AUDIT_REPORT " + str(args.output), flush=True)
        print("AUDIT_STATUS " + report["status"], flush=True)


if __name__ == "__main__":
    main()
