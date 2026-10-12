"""Reversible exact-v8 nasal insertion, preserving original vertex identities.

Input: original v8 arrays before independent neck replacement or later fields.
Output: (vertices, triangles, object_ids, JSON-compatible diagnostics).
Every original vertex row stays unchanged; cut and anatomical vertices append.
Surviving face IDs remap through map_surviving_face_ids(). No selected file is
written by this API. The cache belongs only to the pinned, checked v8 source.
"""
from pathlib import Path
import json
import numpy as np
from stitch_anatomical_patch import stitch_patch

OUT=Path(__file__).resolve().parents[1]/'output/nasal-v8-shared-insertion-study'


def map_surviving_face_ids(face_ids):
    """Exact index map for preserved semantics such as vermilion face IDs."""
    with np.load(OUT/'exact_v8_cut_contract.npz') as q:removed=q['removed_source_face_ids']
    face_ids=np.asarray(face_ids,np.int64);where=np.searchsorted(removed,face_ids)
    if np.any((where<len(removed))&(removed[np.minimum(where,len(removed)-1)]==face_ids)):
        raise ValueError('Requested semantic face is removed by the nasal patch')
    return face_ids-where


def integrate(vertices,triangles,object_ids,*,patch_path=None):
    v=np.asarray(vertices);f=np.asarray(triangles);ids=np.asarray(object_ids)
    meta=json.loads((OUT/'exact_v8_cut_contract.json').read_text())
    if len(v)!=meta['source_vertex_count'] or len(f)!=meta['source_face_count']:
        raise ValueError('Nasal insertion expects original v8 vertex and face order before later replacements')
    patch_path=Path(patch_path) if patch_path is not None else OUT/'fitted_nasal_patch.npz'
    q=np.load(OUT/'exact_v8_cut_contract.npz');p=np.load(patch_path)
    source=q['source_vertex_ids'];source_faces=q['source_crop_face_ids']
    if not np.array_equal(v[source],q['source_positions']):raise ValueError('Original v8 nasal crop coordinates differ; no silent retargeting')
    if not np.array_equal(f[source_faces],source[q['original_crop_triangles']]):raise ValueError('Original v8 local topology/order differs')
    if np.any(ids[source_faces]!=0):raise ValueError('Cached nasal cut would touch a non-skin object')
    removed=q['removed_source_face_ids'];keep=np.ones(len(f),bool);keep[removed]=False
    cut_v=np.r_[v,q['appended_cut_vertices']]
    cut_f=np.r_[f[keep],q['added_retained_triangles']]
    cut_ids=np.r_[ids[keep],q['added_retained_object_ids']]
    out_v,out_f,out_ids,stitch=stitch_patch(cut_v,cut_f,cut_ids,q['boundary_global_vertex_ids'],p['vertices'],p['triangles'],tolerance=1e-12)
    if not np.array_equal(out_v[:len(v)],v):raise AssertionError('Original vertex rows changed')
    report={'source_sha256':meta['source_sha256'],'source_file':meta['source'],'source_original_vertex_count':len(v),'source_original_face_count':len(f),
        'input_verification':'Exact bounded v8 crop coordinates, face identities/order, skin IDs and global array counts; pinned source-file hash was verified when extracting the contract.',
        'original_vertex_map':'Identity for every original row; old interior rows become unreferenced rather than being compacted.',
        'original_positions_unchanged':True,'surviving_original_vertex_ids_preserved':True,
        'cut_vertex_appended_range':[len(v),len(cut_v)],'anatomical_vertex_appended_range':[len(cut_v),len(out_v)],
        'removed_source_faces':len(removed),'retained_face_mapping':'old_id - searchsorted(removed_source_face_ids, old_id), for nonremoved faces; use map_surviving_face_ids()',
        'retained_clip_face_range':[int(keep.sum()),len(cut_f)],'anatomical_face_range':[len(cut_f),len(out_f)],
        'protected_semantic_incident_vertices_or_faces':meta['protected_semantic_incident_vertices_or_faces'],
        'stitch':stitch,'patch_file':str(patch_path),'selected':False,'scope':'Integration diagnostic only. Apply later neck topology and selected global fields afterward.'}
    return out_v,out_f,out_ids,report


def undo(vertices,triangles,object_ids,*,patch_path=None):
    """Restore the exact pre-insertion arrays before any later deformation."""
    patch_path=Path(patch_path) if patch_path is not None else OUT/'fitted_nasal_patch.npz'
    q=np.load(OUT/'exact_v8_cut_contract.npz');p=np.load(patch_path)
    meta=json.loads((OUT/'exact_v8_cut_contract.json').read_text())
    nv=meta['source_vertex_count'];nf=meta['source_face_count'];removed=q['removed_source_face_ids']
    keep=np.ones(nf,bool);keep[removed]=False;cut_end=nv+len(q['appended_cut_vertices']);n=len(q['boundary_global_vertex_ids'])
    patch_map=np.r_[q['boundary_global_vertex_ids'],np.arange(cut_end,cut_end+len(p['vertices'])-n)]
    expected_tail=np.r_[q['added_retained_triangles'],patch_map[p['triangles']]]
    if len(vertices)!=cut_end+len(p['vertices'])-n or len(triangles)!=keep.sum()+len(expected_tail):raise ValueError('Not the untouched insertion result')
    if not np.array_equal(triangles[int(keep.sum()):],expected_tail):raise ValueError('Insertion triangle tail differs')
    if not np.array_equal(vertices[nv:cut_end],q['appended_cut_vertices']) or not np.array_equal(vertices[cut_end:],p['vertices'][n:]):raise ValueError('Inserted geometry has subsequently changed')
    f=np.empty((nf,3),dtype=np.asarray(triangles).dtype);ids=np.empty(nf,dtype=np.asarray(object_ids).dtype)
    f[keep]=triangles[:int(keep.sum())];f[removed]=q['removed_source_triangles']
    ids[keep]=object_ids[:int(keep.sum())];ids[removed]=q['removed_source_object_ids']
    return np.asarray(vertices)[:nv].copy(),f,ids
