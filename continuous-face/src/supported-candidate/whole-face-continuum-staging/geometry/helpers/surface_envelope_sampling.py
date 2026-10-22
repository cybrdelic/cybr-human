"""Area-covering samples of actual exterior triangles for an anatomy envelope.

This fixes a representation error in point-only angular rasterization: a broad
sparse outer fascia triangle must cover the rays through its interior even if
dense deeper anatomy already has vertices in those angular bins. It samples
unaltered selected triangles, without any skin positions or shape fitting.
Finite sample spacing is explicit; this is not exact angular ray casting.
"""
import numpy as np


def root_connective_ribbons(surface, source_part_ids, auricular_vertices, width=8.):
    """Author finite missing connective support from literal native open edges.

    Only boundary edges of the three named lateral connective owners, adjacent
    to a selected outward face and overlapping the auricular support bounds,
    are extended. Each edge is continued in its own source face plane, away
    from the face interior. The returned ribbons are explicitly authored
    support, not recovered anatomy. No skin coordinates are accepted.
    """
    width=float(width)
    if not np.isfinite(width) or width<=0 or width>12:
        raise ValueError('Root connective extension must be in (0, 12] design units')
    av=np.asarray(auricular_vertices,float)
    if av.ndim!=2 or av.shape[1]!=3 or not np.isfinite(av).all():
        raise ValueError('Expected finite authored auricular support vertices')
    v=surface.vertices;faces=surface.faces;part=np.asarray(source_part_ids)
    selected=np.zeros(len(faces),bool);selected[surface.face_ids]=True
    names=('Masseteric fascia','Superficial layer of temporal fascia','Parotid gland')
    triangles=[];edge_ids=[];owner_ids=[];face_ids=[];domains=[]
    for side in (-1,1):
        ear=av[av[:,0]*side>0]
        # The finite source support, rather than a skin/root sculpt, defines
        # the spatial scope. Extend its bounds by the same authored width.
        low=ear.min(0)-width;high=ear.max(0)+width
        domains.append(dict(side=side,minimum=low.tolist(),maximum=high.tolist()))
        for pid,meta in enumerate(surface.manifest['parts']):
            if not meta['name'].startswith(names):continue
            ids=np.flatnonzero(part==pid)
            if not len(ids):continue
            ff=faces[ids]
            edges=np.concatenate([ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]])
            opposite=np.concatenate([ff[:,2],ff[:,0],ff[:,1]])
            face=np.tile(ids,3)
            _,first,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
            rows=first[count==1]
            for row in rows:
                fi=face[row]
                if not selected[fi]:continue
                edge=edges[row];a,b=v[edge];mid=(a+b)*.5
                if mid[0]*side<=0 or np.any(np.maximum(a,b)<low) or np.any(np.minimum(a,b)>high):continue
                along=b-a;length=np.linalg.norm(along)
                if length<1e-9:continue
                along/=length;inside=v[opposite[row]]-mid
                inside-=np.dot(inside,along)*along
                size=np.linalg.norm(inside)
                if size<1e-9:continue
                shift=-width*inside/size
                # Clip all authored vertices to the source-derived local box.
                # Original edge endpoints remain literal source geometry.
                aa=np.clip(a+shift,low,high);bb=np.clip(b+shift,low,high)
                aa=a+(aa-a)*min(1.,width/max(np.linalg.norm(aa-a),1e-12))
                bb=b+(bb-b)*min(1.,width/max(np.linalg.norm(bb-b),1e-12))
                for tri in (np.array([a,aa,bb]),np.array([a,bb,b])):
                    if np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))<1e-10:continue
                    triangles.append(tri);edge_ids.append(edge);owner_ids.append(pid);face_ids.append(fi)
    tri=np.asarray(triangles,float).reshape(-1,3,3)
    packet=dict(triangles=tri,source_boundary_vertex_ids=np.asarray(edge_ids,int).reshape(-1,2),
                source_part_ids=np.asarray(owner_ids,int),source_adjacent_face_ids=np.asarray(face_ids,int))
    report=dict(method='Authored native-open-edge tangent ribbons for missing lateral connective support',
                width_design_units=width,source_edge_count=len(set(map(tuple,np.sort(packet['source_boundary_vertex_ids'],axis=1)))),
                authored_triangles=len(tri),source_owner_names=sorted({surface.manifest['parts'][i]['name'] for i in owner_ids}),
                auricular_support_bounded_domains=domains,source_geometry_unchanged=True,skin_coordinates_used=False,
                affected_bounds=None if not len(tri) else [tri.min((0,1)).tolist(),tri.max((0,1)).tolist()],
                limit='Authored planar boundary continuation, not recovered connective anatomy or a certified solid; overlaps use the same outer radial envelope')
    return packet,report

def triangle_samples(triangles,spacing=.55,point_budget=250000):
    tri=np.asarray(triangles,float)
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(tri).all():
        raise ValueError('Expected finite actual exterior triangles')
    if not np.isfinite(spacing) or spacing<=0 or point_budget<1000:raise ValueError('Invalid sample bounds')
    edge=np.maximum.reduce([np.linalg.norm(tri[:,1]-tri[:,0],axis=1),np.linalg.norm(tri[:,2]-tri[:,1],axis=1),np.linalg.norm(tri[:,0]-tri[:,2],axis=1)])
    levels=np.maximum(1,np.ceil(edge/spacing).astype(int))
    # Group by subdivision count; bounded point batches avoid huge temporary
    # arrays on the current coarse source. No maximum-length face is skipped.
    for level in np.unique(levels):
        face_ids=np.flatnonzero(levels==level)
        i,j=np.meshgrid(np.arange(level+1),np.arange(level+1),indexing='ij');keep=i+j<=level
        b=np.c_[1-(i[keep]+j[keep])/level,i[keep]/level,j[keep]/level]
        batch=max(1,point_budget//len(b))
        for start in range(0,len(face_ids),batch):
            yield np.einsum('pi,nij->npj',b,tri[face_ids[start:start+batch]]).reshape(-1,3)

def raster_outer_radius(triangles,center,shape,spacing=.55,extra_points=None):
    radius=np.full(shape,-np.inf);total=0
    def add(points):
        nonlocal total
        d=np.asarray(points)-center;r=np.linalg.norm(d,axis=1);good=r>1e-10;d=d[good];r=r[good];u=d/r[:,None]
        lat=np.arccos(np.clip(u[:,1],-1,1))*(shape[0]-1)/np.pi
        lon=(np.arctan2(u[:,2],u[:,0])+np.pi)*shape[1]/(2*np.pi)
        ri=np.rint(lat).astype(int);ci=np.rint(lon).astype(int)%shape[1]
        np.maximum.at(radius,(ri,ci),r);total+=len(r)
    for points in triangle_samples(triangles,spacing):add(points)
    if extra_points is not None:add(extra_points)
    return radius,dict(method='Adaptive barycentric sampling of every selected actual exterior triangle',
        maximum_cartesian_sample_step=float(spacing),sample_count=total,
        actual_source_triangles=len(triangles),angular_shape=list(shape),
        angular_coverage=float(np.isfinite(radius).mean()),source_vertices_unchanged=True,
        skin_coordinates_used=False,limit='Finite sampling and later voxel/angular smoothing; not exact ray casting or certified containment')
