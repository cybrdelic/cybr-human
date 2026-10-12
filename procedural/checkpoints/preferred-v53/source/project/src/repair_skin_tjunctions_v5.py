"""Minimal reversible face-only repair for 20 frozen-v5 clipping T-junctions.

No selected mesh is written. Each long-edge face is replaced in its own row
by one child, and the second child is appended. Existing vertex coordinates,
vertex IDs, unaffected face rows and all non-skin objects remain exact.
"""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/skin-boundary-v5-study'
PATCH = OUT/'tjunction_face_repair.npz'
EXPECTED_SHA = 'ba497c73418c1c4e31db3b0b06c07b36656f67c096fc5ce43ea399523e72ea49'


def prepare_patch():
    report = json.loads((OUT/'boundary_localization.json').read_text())
    with np.load(OUT/'boundary_only_context.npz') as q:
        v = q['vertices']; ids = q['source_vertex_ids']; local_f = q['triangles']
        source_faces = q['source_face_ids']
    f = ids[local_f]
    lookup = {int(source): i for i,source in enumerate(ids)}
    changed_rows, before, first, second, records = [], [], [], [], []
    for loop in report['loops']:
        if loop['edges'] != 3: raise ValueError('Only diagnosed three-edge T-junctions are supported')
        original = loop['original_frame']
        if original['distance_to_long_edge'] > 1e-10 or not 0 < original['middle_fraction'] < 1:
            raise ValueError('Original midpoint is not strictly inside the long edge')
        j = original['long_edge_index']; ring = loop['vertex_ids']
        a,b,c = [int(ring[(j+k)%3]) for k in range(3)]
        rows = np.flatnonzero(np.any(f == a,axis=1) & np.any(f == b,axis=1))
        if len(rows) != 1: raise ValueError('Long edge does not have exactly its one boundary face')
        row = int(rows[0]); old = f[row]; k = int(np.flatnonzero(old == a)[0])
        if old[(k+1)%3] != b: raise ValueError('Directed boundary winding mismatch')
        d = int(old[(k+2)%3]); children = np.array([[a,c,d],[c,b,d]])
        coords = np.array([[v[lookup[int(x)]] for x in face] for face in children])
        oldcoords = v[local_f[row]]
        normal = np.cross(oldcoords[1]-oldcoords[0], oldcoords[2]-oldcoords[0])
        cn = np.cross(coords[:,1]-coords[:,0],coords[:,2]-coords[:,0])
        cosine = cn@normal/(np.linalg.norm(cn,axis=1)*np.linalg.norm(normal))
        area = np.linalg.norm(cn,axis=1)/2
        if area.min() <= 1e-12 or cosine.min() <= 0:
            raise ValueError('Child triangle degenerated or reversed')
        changed_rows.append(source_faces[row]); before.append(old); first.append(children[0]); second.append(children[1])
        records.append({'loop':loop['loop'],'source_face':int(source_faces[row]),
                        'long_edge':[a,b],'existing_split_vertex':c,'opposite_vertex':d,
                        'original_midpoint_fraction':original['middle_fraction'],
                        'child_areas':area.tolist(),'child_original_normal_cosines':cosine.tolist()})
    changed_rows = np.asarray(changed_rows,np.int64)
    if len(np.unique(changed_rows)) != 20: raise ValueError('Repairs overlap source face rows')
    np.savez_compressed(PATCH, source_sha256=np.array(EXPECTED_SHA),
                        changed_face_ids=changed_rows, original_faces=np.array(before),
                        replacement_faces=np.array(first), appended_faces=np.array(second),
                        expected_vertex_ids=ids, expected_vertices=v)
    result={'method':'Split each unsplit long-edge neighbor at the existing clipping midpoint',
            'source_sha256':EXPECTED_SHA,'selected_mesh_edited':False,'candidate_type':'Reversible face-edit map only',
            'removed_faces':20,'replacement_faces':40,'net_added_faces':20,
            'vertices_added':0,'vertices_moved':0,'unaffected_face_ids_retained':True,
            'minimum_child_area':float(min(min(r['child_areas']) for r in records)),
            'minimum_child_normal_cosine':float(min(min(r['child_original_normal_cosines']) for r in records)),
            'repairs':records,'global_incidence_dry_run':None,
            'scope':'Candidate face-only topological closure; no render or global self-intersection certificate.'}
    (OUT/'repair_candidate_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['candidate_type','net_added_faces','vertices_moved','minimum_child_area','minimum_child_normal_cosine']},indent=2))


def topology_digest(triangles, object_ids):
    """Canonical digest for this exact face order and object assignment."""
    h=hashlib.sha256(str(tuple(triangles.shape)).encode())
    for start in range(0,len(triangles),100000):
        h.update(np.ascontiguousarray(triangles[start:start+100000],dtype='<i8').tobytes())
        h.update(np.ascontiguousarray(object_ids[start:start+100000],dtype='<i8').tobytes())
    return h.hexdigest()


def apply_candidate(vertices, triangles, object_ids, *, verified_source_sha256=None,
                    allow_same_order_deformation=False, patch_path=PATCH):
    """Return repaired faces/IDs; permit deformed points only with exact topology.

    The same-order option computes the full canonical topology digest rather
    than trusting a caller's statement that these 20 local rows happen to fit.
    """
    with np.load(patch_path) as p:
        rows = p['changed_face_ids']; old = p['original_faces']; new = p['replacement_faces']; extra = p['appended_faces']
        if allow_same_order_deformation:
            if 'source_topology_sha256' not in p: raise ValueError('Patch lacks a validated whole-topology contract')
            if len(vertices) != int(p['source_vertex_count']): raise ValueError('Vertex order/count contract changed')
            if topology_digest(triangles,object_ids) != str(p['source_topology_sha256']):
                raise ValueError('Same-order deformation has different source topology')
        else:
            if verified_source_sha256 != str(p['source_sha256']): raise ValueError('Source SHA not verified against this patch')
            if not np.array_equal(vertices[p['expected_vertex_ids']],p['expected_vertices']): raise ValueError('Boundary coordinates changed')
    if not np.array_equal(triangles[rows],old) or np.any(object_ids[rows] != 0): raise ValueError('Source face IDs or skin material changed')
    parent=vertices[old];children=vertices[np.r_[new,extra]]
    pn=np.cross(parent[:,1]-parent[:,0],parent[:,2]-parent[:,0]);pn=np.r_[pn,pn]
    cn=np.cross(children[:,1]-children[:,0],children[:,2]-children[:,0])
    if np.any(np.linalg.norm(cn,axis=1)<=2e-12) or np.any(np.sum(cn*pn,axis=1)<=0):
        raise ValueError('Current deformation invalidated the local repair orientation')
    faces = np.concatenate([triangles,extra.astype(triangles.dtype)])
    faces[rows] = new
    materials = np.r_[object_ids,np.zeros(len(extra),dtype=object_ids.dtype)]
    return faces,materials


def undo_candidate(triangles, object_ids, *, patch_path=PATCH):
    """Restore the exact original topology while retaining input vertices."""
    with np.load(patch_path) as p:
        rows=p['changed_face_ids']; old=p['original_faces']; new=p['replacement_faces']; extra=p['appended_faces']
    if not np.array_equal(triangles[rows],new) or not np.array_equal(triangles[-len(extra):],extra):
        raise ValueError('Repaired triangle rows changed; cannot undo safely')
    if np.any(object_ids[rows] != 0) or np.any(object_ids[-len(extra):] != 0): raise ValueError('Repaired material IDs changed')
    faces=triangles[:-len(extra)].copy(); faces[rows]=old
    return faces,object_ids[:-len(extra)].copy()


if __name__ == '__main__': prepare_patch()
