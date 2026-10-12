"""Boolean-union the authored original layered routes into a shared closed boundary.
This is geometry, not a volumetric FEM mesh or a new equilibrium model.
"""
from pathlib import Path
import numpy as np,json,hashlib,bpy,bmesh
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];I=B/'route-layout-v8';O=B/'shared-junction-v1';O.mkdir(exist_ok=False);q=np.load(I/'parts.npz');names=['junction','zygomatic_reference','superficial_oral','deep_oral','buccinator_deep','mandibular_depressor'];mats=[]
for n in names:mats.append(bpy.data.materials.new(n))
def obj(name):
 mesh=bpy.data.meshes.new(name);mesh.from_pydata(q[name+'_vertices'].tolist(),[],q[name+'_triangles'].tolist());mesh.update();ob=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(ob)
 for mat in mats:mesh.materials.append(mat)
 for poly in mesh.polygons:poly.material_index=names.index(name)
 return ob
base=obj(names[0]);history=[]
for name in names[1:]:
 other=obj(name);bpy.context.view_layer.objects.active=base;mod=base.modifiers.new('Shared_'+name,'BOOLEAN');mod.operation='UNION';mod.solver='EXACT';mod.object=other;bpy.ops.object.modifier_apply(modifier=mod.name);history.append({'added_region':name,'vertices':len(base.data.vertices),'polygons':len(base.data.polygons)});bpy.data.objects.remove(other,do_unlink=True)
bm=bmesh.new();bm.from_mesh(base.data);bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.to_mesh(base.data);bm.free();mesh=base.data;v=np.array([p.co[:] for p in mesh.vertices],float);f=np.array([p.vertices[:] for p in mesh.polygons],int);regions=np.array([p.material_index for p in mesh.polygons],int)
# Restore double-precision original vertices retained by Boolean. New intersection
# vertices remain explicit construction outputs, with no false original lineage.
source=np.concatenate([q[n+'_vertices'] for n in names]);d,ids=cKDTree(source).query(v);retained=d<5e-5;v[retained]=source[ids[retained]];source_ids=np.where(retained,ids,-1)
# Carry actual bone attachments by exact preserved source positions.
G=np.load(R/'human-preferred-v53-mechanics-20261011/cheek-remesh-v9/geometry.npz');origins={}
for label,points in [('zygomatic',G['vertices'][G['root_node_ids']]),('mandibular',q['mandibular_origin_positions'])]:
 dist,idx=cKDTree(v).query(points);assert dist.max()<1e-10,(label,dist.max());origins[label+'_root_vertex_ids']=idx;origins[label+'_root_positions']=points
origins['mandibular_root_bone_triangle_ids']=q['mandibular_origin_bone_triangle_ids'];origins['mandibular_root_barycentric_weights']=q['mandibular_origin_barycentric_weights'];origins['zygomatic_root_bone_vertex_ids']=G['root_bone_vertex_ids']
# Closed-surface topology and connected components.
edges=np.r_[f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]];_,ic,c=np.unique(np.sort(edges,1),axis=0,return_inverse=True,return_counts=True);wind=np.bincount(ic,weights=np.where(edges[:,0]<edges[:,1],1,-1));parent=np.arange(len(v))
def root(a):
 while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
 return a
for a,b in edges:
 a=root(a);b=root(b)
 if a!=b:parent[b]=a
components=len(set(root(int(i)) for i in np.unique(f)));np.savez_compressed(O/'geometry.npz',vertices=v,triangles=f,region_ids=regions,retained_source_vertex_ids=source_ids,**origins)
r={'input_parts_sha256':hashlib.sha256((I/'parts.npz').read_bytes()).hexdigest(),'geometry_sha256':hashlib.sha256((O/'geometry.npz').read_bytes()).hexdigest(),'region_names':names,'history':history,'vertices':len(v),'triangles':len(f),'components':components,'nonmanifold_edges':int((c!=2).sum()),'inconsistent_winding_edges':int((wind!=0).sum()),'restored_source_vertices':int(retained.sum()),'new_boolean_intersection_vertices':int((~retained).sum()),'zygomatic_roots_exact':len(origins['zygomatic_root_vertex_ids']),'mandibular_roots_exact':len(origins['mandibular_root_vertex_ids']),'limits':['Surface Boolean union only; no tetrahedral domain or coupled solve.','Region IDs retain exterior input-surface provenance, not complete internal fibre/material ownership.','The terminal oral and posterior buccinator cuts remain finite unbound boundaries.']};(O/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));assert components==1 and not (c!=2).any() and not (wind!=0).any()
