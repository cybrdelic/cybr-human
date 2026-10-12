"""Finite inferior entrances on the directly revised supported-tip control net.

The two root curves are obtained from the local metric of the fresh surface.
Every root vertex is shared by exterior and tissue wall. A quarter-round
surface-following inset turns into a curved closed vestibule. There is no old
inlet geometry, independent depth plane, cap over a former hole, or reused
nasal correction field. Units are approximate authored design coordinates.
"""
from functools import lru_cache
import json
import numpy as np
from scipy.spatial import Delaunay
from nasal_fresh_basal_control_net import BOUNDARY,width
from nasal_fresh_supported_base_surface import OUT,evaluate,derivatives

ROOT_CENTER_X=9.7
ROOT_ROW_Y=-26.35
TANGENT_RADII=np.array([2.7,1.9])
ROOT_SAMPLES=80
VESTIBULE_DEPTH=5.2
MIN_CORRESPONDING_DEPTH_RECESS=.28


def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)


@lru_cache(maxsize=2)
def frame(sign):
    uv=np.array([(sign*ROOT_CENTER_X/24+1)/2,(ROOT_ROW_Y+40)/55])
    su,sv=derivatives(uv[None],evaluate);su=su[0];sv=sv[0]
    e1=unit(su);e2=unit(sv-e1*np.dot(e1,sv));normal=unit(np.cross(su,sv))
    # e1/e2 are actual metric axes, not guessed independent opening planes.
    jac=np.stack([su,sv],axis=1)
    metric=np.stack([e1,e2])@jac
    inset_axis=unit(-normal+np.array([0.,0.,-.32]))
    return uv,np.linalg.inv(metric),normal,inset_axis


def parameters(phi,sign,scale=1.):
    uv,inverse,_,_=frame(sign)
    tangent=np.c_[TANGENT_RADII[0]*np.cos(phi),TANGENT_RADII[1]*np.sin(phi)]
    return uv+np.asarray(scale)[...,None]*(tangent@inverse.T)


def metric(uv,sign):
    center,inverse,_,_=frame(sign)
    xy=(np.asarray(uv)-center)@np.linalg.inv(inverse).T
    return np.sum((xy/TANGENT_RADII)**2,axis=1)


def inner_scale(phi):
    # Thicker anterior/superior body and a narrow but finite posterior sill.
    # This varying inset is C-infinity periodic; no separate sector patches.
    return .80-.075*np.sin(phi)


@lru_cache(maxsize=2)
def margin_depth(sign):
    phi=np.linspace(0,2*np.pi,1024,endpoint=False)
    outer=evaluate(parameters(phi,sign));inset=evaluate(parameters(phi,sign,inner_scale(phi)))
    axis=frame(sign)[3]
    # A single smooth depth for the section, determined by corresponding
    # NEW exterior points. No per-vertex clipping or constant-depth plane.
    needed=(inset[:,2]-outer[:,2]+MIN_CORRESPONDING_DEPTH_RECESS)/(-axis[2])
    return max(.8,float(needed.max()))+.025


def rolled_wall(phi,t,sign):
    angle=np.atleast_1d(t)*np.pi/2
    scale=1-(1-inner_scale(phi))[None,:]*np.sin(angle)[:,None]
    depth=margin_depth(sign)*(1-np.cos(angle))
    axis=frame(sign)[3]
    return np.array([evaluate(parameters(phi,sign,s))+d*axis for s,d in zip(scale,depth)])


def vestibule(phi,t,sign):
    inner=rolled_wall(phi,[1.],sign)[0];center=inner.mean(0)
    a=np.atleast_1d(t)*np.pi/2
    axis=frame(sign)[3]
    # Curved superior/posterior continuation with a rounded closed pole.
    # At the shared entrance, its first tangent is exactly the wall axis.
    bend=np.array([0.,-.3,-1.0])
    path=VESTIBULE_DEPTH*np.sin(a)[:,None]*axis+np.sin(a)[:,None]**2*bend
    return center+path[:,None,:]+np.cos(a)[:,None,None]*(inner-center)


def mesh_checks(vertices,triangles,parts):
    p=vertices[triangles];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1)*.5
    if not np.isfinite(vertices).all() or area.min()<1e-10:raise ValueError('Nonfinite or degenerate fresh inlet surface')
    edge=np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]])
    _,inv,count=np.unique(np.sort(edge,axis=1),axis=0,return_inverse=True,return_counts=True)
    oriented=np.where(edge[:,0]<edge[:,1],1,-1)
    winding=np.bincount(inv,weights=oriented)
    if np.any(count>2) or np.any(winding[count==2]!=0):raise ValueError('Nonmanifold or inconsistently oriented fresh inlet')
    return {'minimum_triangle_area':float(area.min()),'nonmanifold_edges':int(np.sum(count>2)),
        'inconsistent_shared_edge_winding':int(np.count_nonzero(winding[count==2])),
        'boundary_edge_count':int(np.sum(count==1)),
        'minimum_part_triangle_areas':{str(k):float(area[parts==k].min()) for k in np.unique(parts)}}


def build_preview(*,boundary_points=None,resolution=(65,81)):
    uv=np.stack(np.meshgrid(np.linspace(0,1,resolution[0]),np.linspace(0,1,resolution[1]),indexing='ij'),-1).reshape(-1,2)
    uv=uv[(metric(uv,1)>1.4)&(metric(uv,-1)>1.4)]
    exact=None
    if boundary_points is not None:
        exact=np.asarray(boundary_points,float)
        if not np.array_equal(exact,BOUNDARY):raise ValueError('Fresh patch only supports the pinned exact attachment order')
        boundary_uv=np.c_[(exact[:,0]/width(exact[:,1])+1)/2,(exact[:,1]+40)/55]
        # Normalize only roundoff in the known rectangular parameter edges.
        # Actual 3D boundary points remain bit-exact, never projected/fitted.
        for value in (0.,1.):boundary_uv[abs(boundary_uv-value)<1e-12]=value
        if np.any(np.min(np.c_[boundary_uv,1-boundary_uv],axis=1)!=0):raise ValueError('Actual attachment is not on the known chart edge')
        interior=uv[(uv.min(1)>0)&(uv.max(1)<1)]
        uv=np.r_[boundary_uv,interior]
    n=ROOT_SAMPLES;phi=np.arange(n)*2*np.pi/n;known=[]
    for sign in (1,-1):
        known.append(np.arange(len(uv),len(uv)+n));uv=np.r_[uv,parameters(phi,sign)]
    faces=Delaunay(uv).simplices;center=uv[faces].mean(1)
    faces=faces[(metric(center,1)>1)&(metric(center,-1)>1)]
    a=uv[faces[:,1]]-uv[faces[:,0]];b=uv[faces[:,2]]-uv[faces[:,0]]
    flip=a[:,0]*b[:,1]-a[:,1]*b[:,0]<0;faces[flip]=faces[flip][:,[0,2,1]]
    edge=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,inv,count=np.unique(np.sort(edge,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=edge[count[inv]==1]
    outer=evaluate(uv)
    if exact is not None:
        if np.max(abs(outer[:len(exact)]-exact))>1e-10:raise ValueError('Fresh evaluator differs from exact attachment')
        outer[:len(exact)]=exact
    blocks=[outer];all_faces=[faces];parts=[np.zeros(len(faces),np.uint8)];cursor=len(outer);reports=[]
    shared_loops=[]
    def bridge(previous,current,part):
        nxt=np.roll(np.arange(n),-1)
        all_faces.extend([np.c_[previous,current[nxt],previous[nxt]],np.c_[previous,current,current[nxt]]])
        parts.extend([np.full(n,part,np.uint8),np.full(n,part,np.uint8)])
    for sign,ids in zip((1,-1),known):
        hole_edges=boundary[np.isin(boundary[:,0],ids)&np.isin(boundary[:,1],ids)]
        if len(hole_edges)!=n:raise ValueError('Exact fresh aperture root not retained by coarse triangulation')
        successor=dict(hole_edges.tolist());loop=[int(ids[0])]
        for _ in range(n-1):loop.append(successor[loop[-1]])
        if successor[loop[-1]]!=loop[0] or len(set(loop))!=n:raise ValueError('Invalid fresh root cycle')
        loop=np.asarray(loop);angle=phi[loop-ids[0]];previous=loop;shared_loops.append(loop)
        wall=rolled_wall(angle,np.linspace(0,1,15),sign)
        if np.max(abs(wall[0]-outer[loop]))>1e-12:raise ValueError('Fresh root position mismatch')
        for ring in wall[1:]:
            current=np.arange(cursor,cursor+n);cursor+=n;blocks.append(ring);bridge(previous,current,1);previous=current
        tube=vestibule(angle,np.linspace(0,1,19),sign)
        if np.max(abs(tube[0]-wall[-1]))>1e-12:raise ValueError('Fresh vestibule position mismatch')
        for ring in tube[1:-1]:
            current=np.arange(cursor,cursor+n);cursor+=n;blocks.append(ring);bridge(previous,current,2);previous=current
        pole=tube[-1].mean(0);blocks.append(pole[None]);pole_id=cursor;cursor+=1
        all_faces.append(np.c_[previous,np.full(n,pole_id),np.roll(previous,-1)]);parts.append(np.full(n,2,np.uint8))
        su,sv=derivatives(parameters(angle,sign),evaluate);normal=unit(np.cross(su,sv))
        reports.append({'side':sign,'shared_root_count':n,'root_world_center':wall[0].mean(0).tolist(),
            'root_world_extent':np.ptp(wall[0],axis=0).tolist(),'inner_world_extent':np.ptp(wall[-1],axis=0).tolist(),
            'center_exterior_normal':frame(sign)[2].tolist(),'root_normal_y_range':[float(normal[:,1].min()),float(normal[:,1].max())],
            'inward_axis':frame(sign)[3].tolist(),'finite_roll_depth':margin_depth(sign),'vestibule_depth':VESTIBULE_DEPTH,
            'minimum_corresponding_z_recess':float((wall[0,:,2]-wall[-1,:,2]).min()),
            'minimum_tissue_roll_displacement':float(np.linalg.norm(wall[-1]-wall[0],axis=1).min()),
            'root_g0_error':float(abs(wall[0]-outer[loop]).max()),'vestibule_g0_error':float(abs(tube[0]-wall[-1]).max())})
    vertices=np.concatenate(blocks);triangles=np.concatenate(all_faces).astype(np.int64);parts=np.concatenate(parts)
    report=mesh_checks(vertices,triangles,parts)
    boundary_uv=np.c_[(BOUNDARY[:,0]/width(BOUNDARY[:,1])+1)/2,(BOUNDARY[:,1]+40)/55]
    error=float(np.max(np.linalg.norm(evaluate(boundary_uv)-BOUNDARY,axis=1)))
    if error>1e-9:raise ValueError('Exact outer attachment position contract changed')
    report.update({'method':'New surface metric frames, exact shared roots, finite tangent-matched rolled tissue, curved closed vestibules',
        'vertices':len(vertices),'triangles':len(triangles),'apertures':reports,'outer_attachment_function_maximum_error':error,
        'old_inlet_or_correction_geometry_reused':False,'fresh_v43_artistic_control_net_directly_revised':True,'coarse_only':exact is None,'integrated':False,
        'exact_boundary_vertices_first':len(exact) if exact is not None else 0,
        'visual_gate_passed':None,'approved_for_integration':False,'approved_for_selection':False})
    return vertices,triangles,parts,uv,np.array(shared_loops),report


if __name__=='__main__':
    vertices,triangles,parts,uv,loops,report=build_preview()
    np.savez_compressed(OUT/'fresh_finite_inlets.npz',vertices=vertices,triangles=triangles,face_parts=parts,exterior_uv=uv,root_vertex_ids=loops)
    (OUT/'fresh_finite_inlets_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
