"""Shared-vertex integration for original authored anatomical patches.

The first N patch vertices are a boundary contract, not duplicate geometry.
They map onto the retained head vertices. Internal patch topology is inspected
before integration; this is not a claim of global self-intersection freedom.
"""
import numpy as np
from scipy.spatial import cKDTree


def stitch_patch(vertices, triangles, object_ids, boundary_indices,
                 patch_vertices, patch_triangles, *, tolerance=2.e-5):
    v=np.asarray(vertices);f=np.asarray(triangles);ids=np.asarray(object_ids)
    boundary=np.asarray(boundary_indices,dtype=np.int64)
    pv=np.asarray(patch_vertices);pf=np.asarray(patch_triangles,dtype=np.int64)
    count=len(boundary)
    if len(pv)<count or pf.min()<0 or pf.max()>=len(pv):
        raise ValueError('Invalid patch vertex/triangle contract')
    distance,nearest=cKDTree(v[boundary]).query(pv[:count])
    if distance.max()>tolerance or len(np.unique(nearest))!=count:
        raise ValueError(f'Patch boundary is not an exact one-to-one fit: {distance.max():.6g}')
    directed_edges=np.concatenate([pf[:,[0,1]],pf[:,[1,2]],pf[:,[2,0]]])
    edges=directed_edges.copy()
    edges.sort(axis=1)
    unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2):
        raise ValueError('Patch contains nonmanifold edges')
    winding=np.bincount(inverse,weights=np.where(directed_edges[:,0]<directed_edges[:,1],1.,-1.))
    if np.any((counts==2)&(winding!=0)):
        raise ValueError('Patch has inconsistent internal triangle winding')
    boundary_edges=unique[counts==1]
    if len(boundary_edges)!=count or boundary_edges.max()>=count:
        raise ValueError('Patch must have exactly its contracted outer boundary')
    mapped=boundary[nearest]
    expected={tuple(sorted((int(a),int(b)))) for a,b in zip(boundary,np.roll(boundary,-1))}
    actual={tuple(sorted((int(mapped[a]),int(mapped[b])))) for a,b in boundary_edges}
    if actual!=expected:
        raise ValueError('Patch boundary edge correspondence differs from the retained mesh')
    mapping=np.r_[mapped,np.arange(len(v),len(v)+len(pv)-count)]
    boundary_mask=np.zeros(len(v),bool);boundary_mask[boundary]=True
    near=f[np.any(boundary_mask[f],axis=1)]
    head_edges={}
    head_incidence={}
    for tri in near:
        for a,b in zip(tri,np.roll(tri,-1)):
            key=tuple(sorted((int(a),int(b))))
            if key in expected:
                head_edges[key]=(int(a),int(b))
                head_incidence[key]=head_incidence.get(key,0)+1
    if len(head_edges)!=count:
        raise ValueError('Retained head does not expose the entire contracted seam')
    if any(q!=1 for q in head_incidence.values()):
        raise ValueError('Contracted head seam includes an internal two-face edge')
    wrong=0
    for a,b in directed_edges[counts[inverse]==1]:
        a,b=int(mapping[a]),int(mapping[b]);key=tuple(sorted((a,b)))
        wrong+=head_edges[key]!=(b,a)
    if wrong:
        raise ValueError(f'Patch winding opposes the required seam orientation at {wrong} edges')
    out_v=np.concatenate([v,pv[count:]])
    out_f=np.concatenate([f,mapping[pf]])
    out_ids=np.r_[ids,np.zeros(len(pf),dtype=ids.dtype)]
    return out_v,out_f,out_ids,{
        'boundary_vertices':count,'maximum_boundary_position_error':float(distance.max()),
        'shared_boundary_edges':len(expected),'patch_triangles':len(pf),
        'patch_boundary_contract_valid':True,
        'scope':'Exact shared positions/edges; global collision and tangent quality require separate checks'}
