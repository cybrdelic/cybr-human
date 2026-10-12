"""Conforming native-v16 oral boundary refinement, without moving old samples.

The original 60-edge oral boundary uses native angles 0, 8, ..., 472. Add the
60 native row-16 samples halfway between them and split only the incident
midface triangles. The returned 120 columns can be used by a stride-4 oral
wedge. Extrude all triangles with the same native-ID ordering rule so the
midface and oral wedge share side-face diagonals through every depth layer.

This helper changes neither the source head nor existing vertex positions. It
does not solve a carrier, choose splits from determinant signs, or certify the
resulting volume. It only constructs exact surface samples and connectivity.
"""

from collections import defaultdict

import numpy as np


def refine_oral_midface_interface(
    vertices,
    native_vertex_ids,
    triangles,
    triangle_parent_faces,
    coarse_oral_angles,
    coarse_oral_columns,
    *,
    external_positions,
    external_vertex_ids,
    mucosal_positions,
    mucosal_vertex_ids,
):
    """Return refined surface arrays and exact 120-column oral correspondence.

    ``vertices`` and ``native_vertex_ids`` describe outer surface columns;
    ``triangles`` index those columns and have already-established outward
    winding. ``triangle_parent_faces`` carries arbitrary integer parent-face
    IDs. ``coarse_oral_angles`` is in native 0..479 units, not 0..59 units, and
    maps one-to-one to ``coarse_oral_columns``. The four native oral chart
    arrays have the same shapes as inputs/native_oral_charts_v16.npz.

    Existing columns keep their indices and values. Returned triangles inherit
    their parent's winding and provenance. New columns, in increasing angular
    order, are exact native exterior row-16 samples. Their carrier endpoints
    are the corresponding exact mucosal row-16 samples. Populate all depth
    nodes once and reuse the returned ``oral_columns`` on both domains.
    """
    vertices = np.asarray(vertices, dtype=float)
    native_vertex_ids = np.asarray(native_vertex_ids, dtype=np.int64)
    triangles = np.asarray(triangles, dtype=np.int64)
    triangle_parent_faces = np.asarray(triangle_parent_faces, dtype=np.int64)
    coarse_oral_angles = np.asarray(coarse_oral_angles, dtype=np.int64)
    coarse_oral_columns = np.asarray(coarse_oral_columns, dtype=np.int64)
    external_positions = np.asarray(external_positions, dtype=float)
    external_vertex_ids = np.asarray(external_vertex_ids, dtype=np.int64)
    mucosal_positions = np.asarray(mucosal_positions, dtype=float)
    mucosal_vertex_ids = np.asarray(mucosal_vertex_ids, dtype=np.int64)

    n = len(vertices)
    if vertices.shape != (n, 3) or native_vertex_ids.shape != (n,):
        raise ValueError("Surface positions/provenance shapes do not agree")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Surface connectivity must contain triangles")
    if triangle_parent_faces.shape != (len(triangles),):
        raise ValueError("Every triangle needs a parent-face ID")
    if np.any(triangles < 0) or np.any(triangles >= n):
        raise ValueError("Surface triangle index is out of range")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Surface positions must be finite")
    if len(np.unique(native_vertex_ids)) != n:
        raise ValueError("Native surface ordering keys must be unique")
    if coarse_oral_angles.shape != (60,) or coarse_oral_columns.shape != (60,):
        raise ValueError("Expected the complete original 60-column oral border")
    if not np.array_equal(np.sort(coarse_oral_angles), np.arange(0, 480, 8)):
        raise ValueError("Coarse angles must be exactly 0, 8, ..., 472")
    if len(np.unique(coarse_oral_columns)) != 60:
        raise ValueError("Coarse oral columns must be distinct")
    if np.any(coarse_oral_columns < 0) or np.any(coarse_oral_columns >= n):
        raise ValueError("Coarse oral column is out of range")
    for positions in (external_positions, mucosal_positions):
        if positions.shape != (17, 480, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("Expected finite native oral positions, shape 17x480x3")
    for ids in (external_vertex_ids, mucosal_vertex_ids):
        if ids.shape != (17, 480):
            raise ValueError("Expected native oral provenance, shape 17x480")
    if not np.array_equal(
        vertices[coarse_oral_columns], external_positions[16, coarse_oral_angles]
    ):
        raise ValueError("Existing oral exterior does not match exact native samples")
    if not np.array_equal(
        native_vertex_ids[coarse_oral_columns],
        external_vertex_ids[16, coarse_oral_angles],
    ):
        raise ValueError("Existing oral provenance does not match the native chart")

    border = dict(zip(coarse_oral_angles.tolist(), coarse_oral_columns.tolist()))
    edge_owners = defaultdict(list)
    for ti, triangle in enumerate(triangles):
        if len(set(triangle.tolist())) != 3:
            raise ValueError("Degenerate surface triangle connectivity")
        for first, second in zip(triangle, np.roll(triangle, -1)):
            edge_owners[tuple(sorted((int(first), int(second))))].append(ti)

    midpoint_angles = np.arange(4, 480, 8, dtype=np.int64)
    new_columns = np.arange(n, n + 60, dtype=np.int64)
    new_native_ids = external_vertex_ids[16, midpoint_angles]
    if np.intersect1d(native_vertex_ids, new_native_ids).size:
        raise ValueError("A proposed midpoint already exists as a surface column")
    if len(np.unique(new_native_ids)) != 60:
        raise ValueError("Midpoint native IDs are not unique")
    split = {}
    for angle, column in zip(midpoint_angles, new_columns):
        first = border[int(angle - 4)]
        second = border[int((angle + 4) % 480)]
        edge = tuple(sorted((first, second)))
        owners = edge_owners[edge]
        if len(owners) != 1:
            raise ValueError("Every oral edge must have exactly one midface triangle")
        ti = owners[0]
        if ti in split:
            raise ValueError("Cannot split two oral edges of one incident triangle")
        split[ti] = (edge, int(column))
        border[int(angle)] = int(column)

    refined_triangles = []
    source_indices = []
    for ti, triangle in enumerate(triangles):
        if ti not in split:
            refined_triangles.append(triangle.tolist())
            source_indices.append(ti)
            continue
        edge, midpoint = split[ti]
        for shift in range(3):
            first, second, third = np.roll(triangle, -shift).tolist()
            if tuple(sorted((first, second))) == edge:
                # Cyclic rotation followed by a directed-edge split preserves
                # winding. The native midpoint need not be the chord midpoint.
                refined_triangles.extend(
                    [[first, midpoint, third], [midpoint, second, third]]
                )
                source_indices.extend([ti, ti])
                break
        else:
            raise AssertionError("Recorded oral edge was not in its owner triangle")

    source_indices = np.asarray(source_indices, dtype=np.int64)
    oral_angles = np.arange(0, 480, 4, dtype=np.int64)
    return {
        "vertices": np.concatenate([vertices, external_positions[16, midpoint_angles]]),
        "native_vertex_ids": np.concatenate([native_vertex_ids, new_native_ids]),
        "triangles": np.asarray(refined_triangles, dtype=np.int64),
        "triangle_parent_faces": triangle_parent_faces[source_indices].copy(),
        "triangle_source_indices": source_indices,
        "oral_angles": oral_angles,
        "oral_columns": np.asarray([border[int(a)] for a in oral_angles], dtype=np.int64),
        "oral_inner_positions": mucosal_positions[16, oral_angles].copy(),
        "oral_mucosal_native_ids": mucosal_vertex_ids[16, oral_angles].copy(),
        "new_columns": new_columns,
        "new_inner_displacements": (
            mucosal_positions[16, midpoint_angles]
            - external_positions[16, midpoint_angles]
        ),
        "split_triangle_indices": np.asarray(sorted(split), dtype=np.int64),
    }
