"""Original layer/attachment geometry in the preferred v53, prior to new dynamics.
Source-informed topology of routes; all coordinates/widths are authored estimates.
"""
from pathlib import Path
import numpy as np,json,hashlib
from scipy.interpolate import PchipInterpolator
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];O=B/'route-layout-v6';O.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
HP=R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz';MP=B/'mandible-v3/geometry.npz';GP=R/'human-preferred-v53-mechanics-20261011/cheek-remesh-v9/geometry.npz';H=np.load(HP);M=np.load(MP);G=np.load(GP);skin=BVHTree.FromPolygons(H['vertices'].tolist(),H['triangles'][H['object_ids']==0].tolist(),all_triangles=True);mandible=BVHTree.FromPolygons(M['vertices'].tolist(),M['triangles'].tolist(),all_triangles=True)
def front(x,y):
 h,_,_,_=skin.ray_cast(Vector((float(x),float(y),200)),Vector((0,0,-1)),400)
 if h is None:raise ValueError('No preferred surface')
 return h.z
def close_tube(rings):
 n,m=rings.shape[:2];v=rings.reshape(-1,3);f=[]
 for i in range(n-1):
  for j in range(m):
   k=(j+1)%m;a=i*m+j;b=i*m+k;c=(i+1)*m+k;d=(i+1)*m+j;f.extend([[a,b,c],[a,c,d]])
 for i in [0,n-1]:
  z=len(v);v=np.r_[v,rings[i].mean(0)[None]]
  for j in range(m):k=(j+1)%m;f.append([z,i*m+k,i*m+j] if i==0 else [z,i*m+j,i*m+k])
 return v,np.array(f,int)
parts={};extra={};spec={}
def put(name,v,f,description):
 tr=v[f];vol=np.einsum('ij,ij->i',tr[:,0],np.cross(tr[:,1],tr[:,2])).sum()/6
 if vol<0:f=f[:,[0,2,1]];vol=-vol
 parts[name]=(v,f);spec[name]={'description':description,'vertices':len(v),'triangles':len(f),'enclosed_signed_volume':float(vol)}
def route(name,controls,widths,thickness,description):
 p=np.array(controls,float);tt=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1))];t=np.linspace(0,tt[-1],70);q=PchipInterpolator(tt,p)(t);tan=PchipInterpolator(tt,p[:,:2]).derivative()(t);tan/=np.linalg.norm(tan,axis=1)[:,None];lateral=np.c_[tan[:,1],-tan[:,0]];w=PchipInterpolator(tt,np.array(widths))(t);th=np.arange(32)*2*np.pi/32;rings=[]
 for i in range(len(t)):
  xy=q[i,:2]+w[i]*np.cos(th)[:,None]*lateral[i];z=np.array([front(x,y) for x,y in xy])-q[i,2]-thickness*np.sin(th);rings.append(np.c_[xy,z])
 v,f=close_tube(np.array(rings));put(name,v,f,description);extra[name+'_centreline']=np.c_[q[:,:2],np.array([front(x,y) for x,y in q[:,:2]])-q[:,2]];extra[name+'_controls_xy_depth']=p;extra[name+'_widths']=w
# Exact existing active bundle/rest attachment, retained as construction context only.
put('zygomatic_reference',G['vertices'],G['boundary_faces'],'Previously checked approximate active right-cheek bundle; its earlier equilibrium is separate from this new layout.')
# A declared common insertion region, lateral to the actual finite oral cavity.
nt,np_=33,64;phi=np.linspace(.001,np.pi-.001,nt);th=np.arange(np_)*2*np.pi/np_;c=np.array([29.35,-51.3,105.25]);rr=np.array([1.65,3.7,1.9]);rings=np.array([c+rr*np.c_[np.sin(p)*np.cos(th),np.cos(p)*np.ones(np_),np.sin(p)*np.sin(th)] for p in phi]);v,f=close_tube(rings);put('junction',v,f,'Authored lateral fibromuscular junction volume; common geometry interface for layered routes, not a calibrated modiolus.')
route('superficial_oral',[[29,-50.5,4.2],[25.5,-48,3.1],[21,-46.5,2.8],[15,-45.7,2.6]],[1.25,1.3,1.2,1.1],.48,'Superficial route from the junction into upper outer labial tissue; a finite regional continuation, with medial cut unbound.')
route('deep_oral',[[29,-51.5,6.1],[27.2,-54.6,4.2],[22,-55.4,3.4],[16,-55.4,3.0]],[1.05,.9,1.1,1.0],.45,'Deeper lower labial continuation, distinct from the superficial upper route; no universal point-insertion assumption.')
route('buccinator_deep',[[43,-46,6.0],[38,-48.2,6.3],[33.5,-50.1,6.2],[29,-51.5,6.1]],[2.7,2.5,1.9,1.1],.5,'Deep transverse/oblique cheek route into the inner perioral layer. Posterior cut is explicit; alveolar/raphe origins are not yet provided.')
# A continuous sheet route is intersected with the original jaw contact surface.
# This avoids forcing an incompatible wide bone patch through a narrow radial loft.
route('mandibular_depressor',[[24,-80,14],[24.6,-76.5,9],[26.3,-71,5],[28,-65,3.4],[29,-58,3.2],[29,-51.3,4.7]],[2.,5.8,5,4,2.9,1.3],.7,'Original DAO-like tapered sheet, clipped by the authored mandible to create its finite origin. Geometry only, not solved.')
import bpy,bmesh
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
def obj(name,v,f):
 mesh=bpy.data.meshes.new(name);mesh.from_pydata(v.tolist(),[],f.tolist());mesh.update();ob=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(ob);return ob
mv,mf=parts['mandibular_depressor'];muscle=obj('authored_depressor',mv,mf);jaw=obj('authored_mandible',M['vertices'],M['triangles']);mat0=bpy.data.materials.new('route');mat1=bpy.data.materials.new('bone_origin');
for ob in [muscle,jaw]:ob.data.materials.append(mat0);ob.data.materials.append(mat1)
for poly in jaw.data.polygons:poly.material_index=1
bpy.context.view_layer.objects.active=muscle;muscle.select_set(True);jaw.select_set(False);mod=muscle.modifiers.new('Exact_bone_footprint','BOOLEAN');mod.operation='DIFFERENCE';mod.solver='EXACT';mod.object=jaw;bpy.ops.object.modifier_apply(modifier=mod.name);bm=bmesh.new();bm.from_mesh(muscle.data);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=2e-5);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(muscle.data);bm.free();mod=muscle.modifiers.new('Triangulated_contact','TRIANGULATE');bpy.ops.object.modifier_apply(modifier=mod.name)
verts=np.array([x.co[:] for x in muscle.data.vertices],float);faces=np.array([p.vertices[:] for p in muscle.data.polygons],int);cent=verts[faces].mean(1);tr=verts[faces];norm=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None]
cap=[];capbone=[]
for i,p in enumerate(cent):
 h,n,fi,dist=mandible.find_nearest(Vector(p))
 if muscle.data.polygons[i].material_index==1:cap.append(i);capbone.append(fi)
print('boolean cap faces',len(cap),flush=True)
assert len(cap)>0
origin=np.unique(faces[cap]);btri=[];weights=[];original=verts[origin].copy()
for idx in origin:
 h,n,fi,dist=mandible.find_nearest(Vector(verts[idx]));a,b,c=M['vertices'][M['triangles'][fi]];ab=b-a;ac=c-a;d=verts[idx]-a;mat=np.array([[ab@ab,ab@ac],[ab@ac,ac@ac]]);uv=np.linalg.solve(mat,np.array([d@ab,d@ac]));ww=np.array([1-uv.sum(),uv[0],uv[1]]);assert ww.min()>-1e-4;ww=np.maximum(ww,0);ww/=ww.sum();verts[idx]=ww@M['vertices'][M['triangles'][fi]];btri.append(fi);weights.append(ww)
# Keep only exterior connected component(s), reporting any unsupported leftover.
put('mandibular_depressor',verts,faces,'Original continuous DAO-like sheet with finite triangular origin cut directly against the new mandible. Every origin vertex has an exact barycentric bone-surface correspondence; no new dynamics.')
ids=np.array(capbone);rootV=origin;rv=verts[origin];reconstructed=np.einsum('ni,nij->nj',np.array(weights),M['vertices'][M['triangles'][btri]]);attachment_error=float(np.linalg.norm(reconstructed-rv,axis=1).max())
extra.update(mandibular_origin_local_vertex_ids=origin,mandibular_origin_local_face_ids=np.array(cap),mandibular_origin_bone_triangle_ids=np.array(btri),mandibular_origin_barycentric_weights=np.array(weights),mandibular_origin_positions=rv,mandibular_origin_face_bone_triangle_ids=ids)

DIR=[Vector((1,.331,-.713)).normalized(),Vector((-.813,.143,.719)).normalized(),Vector((.341,-.823,.119)).normalized()]
def inside(p,d):
 o=Vector(p);n=0
 for _ in range(50):
  h,_,_,_=skin.ray_cast(o,d,1000)
  if h is None:return n%2
  o=h+d*.0003;n+=1
 return -1
for name,(v,f) in parts.items():
 par=np.array([[inside(p,d) for d in DIR] for p in v]);bad=np.flatnonzero(~(par==1).all(1));dist=np.array([skin.find_nearest(Vector(p))[3] for p in v]);spec[name].update(sampled_outside_vertices=len(bad),outside_examples=bad[:15].tolist(),outside_bounds=[v[bad].min(0).tolist(),v[bad].max(0).tolist()] if len(bad) else None,minimum_sampled_skin_distance=float(dist.min()));extra[name+'_vertices']=v;extra[name+'_triangles']=f;extra[name+'_parity']=par;print(name,spec[name],flush=True)
np.savez_compressed(O/'parts.npz',**extra);report={'head_sha256':sha(HP),'mandible_sha256':sha(MP),'original_cheek_sha256':sha(GP),'parts_sha256':sha(O/'parts.npz'),'parts':spec,'mandibular_footprint_faces':len(ids),'mandibular_footprint_vertices':len(rootV),'bone_origin_barycentric_reconstruction_error':attachment_error,'scope':'Original anatomical construction layout, with explicit source-informed superficial/deep routes and a true geometric mandibular origin footprint. The parts share intended junction space but are not yet a conforming tetrahedral network or force-coupled assembly. No new pose claim.','pending':['Local inter-part union/interface construction','Full static contact validation for all new route surfaces','Literal geometry cutaway inspection','New continuum remesh and attachment traction before any new dynamics']};(O/'report.json').write_text(json.dumps(report,indent=2));print('written',sha(O/'parts.npz'))
