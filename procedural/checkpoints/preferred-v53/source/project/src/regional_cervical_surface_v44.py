"""Original editable regional chin, neck and shoulder surface, v44 muscle-lane study.

Cuts a single regular contour from our procedural skin, preserves actual
shared edge positions/tangents, and creates a C2-faired three-dimensional
regional guide surface. No pre-authored head, scan or image texture is used.
The controls are artistic design units, not calibrated anatomical measures.
This candidate moves the cut above the prior jaw/neck ridge so the connected
regional surface can span the transition rather than inherit its old position.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline,BSpline
AXIS_Z=20.
BASE_Y=-181.
MUSCLE_START=1.9
TOP_ANGLES=np.array([0.0, 0.5235987755982988, 1.0471975511965976, 1.5707963267948966, 2.0943951023931953, 3.141592653589793],dtype=float)
TOP_Y=np.array([-76.0, -72.0, -64.0, -58.0, -57.0, -57.0],dtype=float)
AX=np.array([0.0, 0.25, 0.5, 0.8, 1.2, 1.5707963267948966, 2.1, 2.6, 3.141592653589793],dtype=float)
LEVELS=np.array([0.0, 0.1, 0.22, 0.38, 0.6, 0.8, 1.0],dtype=float)
RADIUS=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [96.0, 93.0, 76.0, 61.0, 52.0, 52.0, 58.0, 64.0, 68.0], [80.0, 80.0, 67.0, 51.0, 46.0, 48.0, 59.0, 66.0, 69.0], [64.0, 64.0, 61.0, 55.0, 49.0, 47.0, 56.0, 68.0, 72.0], [62.0, 64.0, 67.0, 79.0, 93.0, 98.0, 88.0, 82.0, 89.0], [63.0, 71.0, 89.0, 121.0, 143.0, 147.0, 126.0, 104.0, 102.0], [66.0, 73.0, 98.0, 139.0, 154.0, 158.0, 139.0, 114.0, 104.0]],dtype=float)
YANGLES=np.array([0.0, 0.5, 1.1, 1.5707963267948966, 2.2, 3.141592653589793],dtype=float)
YCONTROL=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-94.0, -89.0, -81.0, -78.0, -79.0, -79.0], [-103.0, -99.0, -93.0, -92.0, -97.0, -99.0], [-120.0, -122.0, -124.0, -124.0, -123.0, -122.0], [-148.0, -149.0, -147.0, -145.0, -146.0, -147.0], [-167.0, -168.0, -168.0, -166.0, -166.0, -166.0], [-181.0, -181.0, -181.0, -181.0, -181.0, -181.0]],dtype=float)


def contour_scalar(v):
 theta=np.arctan2(v[:,0],v[:,2]-AXIS_Z)
 return v[:,1]-PchipInterpolator(TOP_ANGLES,TOP_Y)(np.abs(theta))

def cut_upper_skin(v,f,ids):
 s=contour_scalar(v);skin=ids==0;sf=s[f]
 crossing=skin&(sf.min(1)<0)&(sf.max(1)>0)
 keep=(~skin)|(sf.min(1)>=0)
 # Actual area-weighted local normals around the contour, not a guessed
 # constant collar normal. All incident faces are included in the band.
 band=np.abs(s)<5.;near=f[skin&np.any(band[f],axis=1)]
 p=v[near];nn=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normals=np.zeros_like(v)
 for j in range(3):np.add.at(normals,near[:,j],nn)
 normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-14)
 vertices=[q.copy() for q in v] if False else None
 extra=[];extra_n=[];cache={};contour=[];clipped=[]
 def edge(a,b):
  key=tuple(sorted((int(a),int(b))))
  if key not in cache:
   a,b=key;t=s[a]/(s[a]-s[b]);point=(1-t)*v[a]+t*v[b];normal=(1-t)*normals[a]+t*normals[b];normal/=np.linalg.norm(normal)
   cache[key]=len(v)+len(extra);extra.append(point);extra_n.append(normal)
  return cache[key]
 for tri in f[crossing]:
  poly=[];cuts=[]
  for i in range(3):
   a=int(tri[i]);b=int(tri[(i+1)%3]);ina=s[a]>=0;inb=s[b]>=0
   if ina:poly.append(a)
   if ina!=inb:
    k=edge(a,b);poly.append(k);cuts.append(k)
  if len(cuts)!=2:raise ValueError('Nonregular cut triangle')
  contour.append(cuts)
  for j in range(1,len(poly)-1):clipped.append([poly[0],poly[j],poly[j+1]])
 adj={}
 for a,b in contour:adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
 if any(len(q)!=2 for q in adj.values()):raise ValueError('Cut does not form closed regular cycles')
 first=next(iter(adj));loop=[first];prev=None;cur=first
 while True:
  nxt=adj[cur][0] if adj[cur][0]!=prev else adj[cur][1]
  if nxt==first:break
  if nxt in loop:raise ValueError('Non-simple cut cycle')
  loop.append(nxt);prev,cur=cur,nxt
 if len(loop)!=len(adj):raise ValueError('More than one cut cycle; no component discarded')
 extra=np.array(extra);extra_n=np.array(extra_n);allv=np.r_[v,extra];loop=np.array(loop)
 theta=np.mod(np.arctan2(allv[loop,0],allv[loop,2]-AXIS_Z),2*np.pi)
 order=np.argsort(theta);loop=loop[order];theta=theta[order]
 # The cut must be genuinely star-shaped in angular order before this
 # periodic neck chart can be used. Check graph adjacency, not only a sort.
 if any(int(loop[(i+1)%len(loop)]) not in adj[int(loop[i])] for i in range(len(loop))):raise ValueError('Cut boundary is not star-shaped')
 if np.min(np.diff(np.r_[theta,theta[0]+2*np.pi]))<1e-8:raise ValueError('Repeated angular cut parameters')
 headf=np.r_[f[keep],np.array(clipped,np.int64)];headids=np.r_[ids[keep],np.zeros(len(clipped),ids.dtype)]
 return allv,headf,headids,loop,theta,extra_n[loop-len(v)],{'removed_lower_skin_faces':int(np.count_nonzero(skin&~keep)),'new_cut_vertices':len(extra),'shared_boundary_vertices':len(loop),'kept_original_head_face_vertices_unchanged':True}

def periodic_interp(theta,source_theta,values):
 theta=np.mod(theta,2*np.pi);xx=np.r_[source_theta[-1]-2*np.pi,source_theta,source_theta[0]+2*np.pi];yy=np.concatenate([values[-1:],values,values[:1]])
 return np.stack([np.interp(theta,xx,yy[:,axis]) for axis in range(values.shape[1])],axis=1)

def guide_surface(theta,t,top_theta,top,top_normals):
 theta=np.asarray(theta);t=np.atleast_1d(t);angle=np.abs(np.angle(np.exp(1j*theta)))
 b=periodic_interp(theta,top_theta,top);bn=periodic_interp(theta,top_theta,top_normals);bn/=np.linalg.norm(bn,axis=1,keepdims=True)
 si=np.sin(theta);co=np.cos(theta);controls=[b]
 for j,level in enumerate(LEVELS[1:],1):
  rr=CubicSpline(AX,RADIUS[j],bc_type=((1,0.),(1,0.)))(angle)
  y=CubicSpline(YANGLES,YCONTROL[j],bc_type=((1,0.),(1,0.)))(angle)
  start=MUSCLE_START
  center=start-(level**.85)*(start-.31);distance=abs(angle-center)
  # Muscle lanes are evaluated on the completed surface below, not blurred
  # away as isolated control-row radius additions.
  controls.append(np.c_[rr*si,y,20+rr*co])
 controls=np.stack(controls)
 # Direction from the measured boundary normal, downward speed from the
 # first authored section. Exact tangent plane, no ad hoc collar blend.
 down=np.zeros_like(b);down[:,1]=-1
 tangent=down-np.sum(down*bn,axis=1)[:,None]*bn
 speed=(b[:,1]-controls[1,:,1])/LEVELS[1]
 tangent*= (speed/np.maximum(-tangent[:,1],.25))[:,None]
 knots=np.r_[[0.]*4,[.11,.24,.40,.60,.80],[1.]*4];count=len(knots)-4
 basis=BSpline(knots,np.eye(count),3)
 B=basis(LEVELS);grid=np.linspace(0,1,121);D=basis.derivative(2)(grid)
 weights=np.array([0.,.5,2.,1.,1.,1.,0.])
 # Soft section fit plus integrated bending; only the actual shared edge,
 # its tangent, and the base silhouette are equality constraints.
 H=B.T@(weights[:,None]*B)+.000035*(D.T@D)/len(grid)
 C=np.stack([basis(0),basis.derivative()(0),basis(1)])
 K=np.block([[H,C.T],[C,np.zeros((3,3))]])
 rhs=np.concatenate([B.T@(weights[:,None]*controls.reshape(len(LEVELS),-1)),np.stack([b,tangent,controls[-1]]).reshape(3,-1)],axis=0)
 coeff=np.linalg.solve(K,rhs)[:count].reshape(count,len(theta),3)
 result=np.einsum('ti,iaj->taj',basis(t),coeff)
 # Broad SCM surface lanes travel from the posterior mastoid direction to
 # distinct sternal and clavicular insertions. Their positions follow the
 # authored 3D neck chart, and their thickness vanishes with zero endpoint
 # derivative at both the real shared boundary and the shoulder base.
 center=PchipInterpolator([0,.10,.22,.38,.60,.80,1.],[1.90,1.45,1.10,.64,.28,.18,.17])(t)
 height=PchipInterpolator([0,.06,.18,.34,.55,.80,.94,1.],[0,0,1.4,2.5,2.2,.6,0,0])(t)
 width=PchipInterpolator([0,.20,.45,.7,1.],[.26,.25,.23,.20,.18])(t)
 def lane(c,wd):
  u=(angle[None,:]-c[:,None])/wd[:,None]
  return np.maximum(0,1-u*u)**3
 main=height[:,None]*lane(center,width)
 split=PchipInterpolator([0,.38,.52,.72,.88,1.],[0,0,.65,1.4,.5,0])(t)
 clav_center=center+.36*np.clip((t-.34)/.3,0,1)
 branch=split[:,None]*lane(clav_center,width*1.10)
 relief=main+branch
 radial=result.copy();radial[:,:,1]=0;radial[:,:,2]-=AXIS_Z
 radial/=np.maximum(np.linalg.norm(radial,axis=2,keepdims=True),1e-15)
 result+=relief[:,:,None]*radial
 result[np.isclose(t,0)]=b
 return result,controls

def build(v,f,ids,m=256,rows=88):
 allv,headf,headids,loop,theta,normals,report=cut_upper_skin(v,f,ids);top=allv[loop];n=len(loop)
 regular_theta=np.arange(m)*2*np.pi/m;ts=np.r_[.010,np.linspace(.024,1,rows)]
 first,_=guide_surface(theta,[ts[0]],theta,top,normals);regular,controls=guide_surface(regular_theta,ts[1:],theta,top,normals)
 parts=[allv,first[0],regular.reshape(-1,3)];cursor=len(allv);first_ids=np.arange(cursor,cursor+n);cursor+=n;regular_ids=np.arange(cursor,cursor+regular.size//3).reshape(len(regular),m);cursor+=regular.size//3
 faces=[]
 def connect_equal(a,b):
  for i in range(len(a)):
   j=(i+1)%len(a);faces.extend([[a[i],b[i],b[j]],[a[i],b[j],a[j]]])
 connect_equal(loop,first_ids)
 # Exact unequal-density shared zipper. Both angular sequences use the
 # same cut seam, so no point snapping or unmatched border is introduced.
 seam=theta[0];rt=seam+np.arange(m)*2*np.pi/m
 regular,controls=guide_surface(rt,ts[1:],theta,top,normals);parts[-1]=regular.reshape(-1,3)
 ta=np.r_[theta,theta[0]+2*np.pi];tb=np.r_[rt,rt[0]+2*np.pi];ia=ib=0
 while ia<n or ib<m:
  a=first_ids[ia%n];b=regular_ids[0,ib%m];na=ta[ia+1] if ia<n else np.inf;nb=tb[ib+1] if ib<m else np.inf
  if na<=nb:faces.append([a,b,first_ids[(ia+1)%n]]);ia+=1
  else:faces.append([a,b,regular_ids[0,(ib+1)%m]]);ib+=1
 for a,b in zip(regular_ids[:-1],regular_ids[1:]):connect_equal(a,b)
 center=np.array([0.,BASE_Y,5.]);parts.append(center[None]);last=regular_ids[-1]
 for i in range(m):faces.append([last[i],cursor,last[(i+1)%m]])
 vertices=np.concatenate(parts);newf=np.asarray(faces,np.int64);triangles=np.r_[headf,newf];object_ids=np.r_[headids,np.zeros(len(newf),headids.dtype)]
 # Full directed skin-edge incidence verifies the actual weld, not just
 # coincident boundary coordinates. This is not a self-intersection proof.
 sf=triangles[object_ids==0];edges=np.concatenate([sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]]]);a=np.minimum(edges[:,0],edges[:,1]);b=np.maximum(edges[:,0],edges[:,1]);code=a.astype(np.uint64)*np.uint64(len(vertices))+b.astype(np.uint64);order=np.argsort(code);codes=code[order];starts=np.r_[0,np.flatnonzero(np.diff(codes))+1];counts=np.diff(np.r_[starts,len(codes)]);direct=np.where(edges[:,0]<edges[:,1],1,-1)[order];balance=np.add.reduceat(direct,starts)
 report.update({'new_surface_vertices':int(sum(len(q) for q in parts[1:])),'new_surface_triangles':len(newf),'skin_boundary_edges':int(np.count_nonzero(counts==1)),'skin_nonmanifold_edges':int(np.count_nonzero(counts!=2)),'directed_edge_imbalances':int(np.count_nonzero(balance)),'topology_scope':'Full skin directed incidence only; global self-intersections are not certified.','base_y':BASE_Y})
 if np.any(counts!=2) or np.any(balance):raise ValueError(report)
 return vertices,triangles,object_ids,report,{'top':top,'top_normals':normals,'top_theta':theta,'guide_controls':controls}
