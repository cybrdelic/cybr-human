"""Use the already installed Gmsh 4.13 shared library via its C API.
No downloaded geometry or package installation. Exact source skin is untouched.
The checked union triangles are supplied as an existing discrete boundary.
"""
from pathlib import Path
import ctypes as C,numpy as np,json,hashlib
B=Path(__file__).resolve().parents[1];I=B/'shared-junction-v2';O=B/'shared-tet-v4';O.mkdir(exist_ok=False);q=np.load(I/'geometry.npz');v=q['vertices'];f=q['triangles'];msh=O/'boundary.msh'
with open(msh,'w') as s:
 s.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n'%len(v))
 for i,p in enumerate(v,1):s.write('%d %.17g %.17g %.17g\n'%(i,*p))
 s.write('$EndNodes\n$Elements\n%d\n'%len(f))
 for i,t in enumerate(f+1,1):s.write('%d 2 2 1 1 %d %d %d\n'%(i,*t))
 s.write('$EndElements\n')
geo=O/'domain.geo';geo.write_text('Merge "'+str(msh)+'";\nSurface Loop(1) = {1};\nVolume(1) = {1};\n')
lib=C.CDLL('/usr/lib/x86_64-linux-gnu/libgmsh.so.4.13');err=C.c_int()
def call(n,*args):
 err.value=0;getattr(lib,n)(*args,C.byref(err))
 if err.value:raise RuntimeError(n+' error '+str(err.value))
call('gmshInitialize',C.c_int(0),C.c_void_p(),C.c_int(0),C.c_int(0))
def option(k,x):call('gmshOptionSetNumber',C.c_char_p(k.encode()),C.c_double(x))
for k,x in [('General.Terminal',1),('General.NumThreads',2),('Mesh.MaxNumThreads3D',2),('Mesh.MeshOnlyEmpty',1),('Mesh.Algorithm3D',10),('Mesh.Optimize',1),('Mesh.OptimizeNetgen',0),('Mesh.MeshSizeMin',.12),('Mesh.MeshSizeMax',1.0),('Mesh.MshFileVersion',2.2),('Mesh.SaveAll',1)]:option(k,x)
call('gmshOpen',C.c_char_p(str(geo).encode()));call('gmshModelMeshGenerate',C.c_int(3));output=O/'tetrahedra.msh';call('gmshWrite',C.c_char_p(str(output).encode()));call('gmshFinalize')
lines=output.read_text().splitlines();k=lines.index('$Nodes');n=int(lines[k+1]);data=np.array([[float(x) for x in row.split()] for row in lines[k+2:k+2+n]]);ids=data[:,0].astype(int);vertices=data[:,1:];lookup={int(k):i for i,k in enumerate(ids)};k=lines.index('$Elements');ne=int(lines[k+1]);tets=[];surface=[]
for row in lines[k+2:k+2+ne]:
 a=[int(x) for x in row.split()];typ=a[1];tags=a[2];nodes=a[3+tags:]
 if typ==4:tets.append([lookup[x] for x in nodes])
 if typ==2:surface.append([lookup[x] for x in nodes])
tets=np.array(tets,int);surface=np.array(surface,int);assert len(tets)>0
p=vertices[tets];det=np.linalg.det(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]],axis=-1));negative=det<0;tets[negative]=tets[negative][:,[0,2,1,3]];det=np.abs(det)
from scipy.spatial import cKDTree
d,match=cKDTree(vertices).query(v);assert d.max()<1e-10;boundary_original=np.sort(match[f],axis=1);given=np.sort(surface,axis=1);aset={tuple(x) for x in boundary_original};bset={tuple(x) for x in given};np.savez_compressed(O/'geometry.npz',vertices=vertices,tetrahedra=tets,boundary_faces=surface,source_surface_vertex_ids=match,**{k:match[q[k]] for k in ['zygomatic_root_vertex_ids','mandibular_root_vertex_ids']})
r={'source_sha256':hashlib.sha256((I/'geometry.npz').read_bytes()).hexdigest(),'geometry_sha256':hashlib.sha256((O/'geometry.npz').read_bytes()).hexdigest(),'mesher':'Preinstalled Gmsh4.13 system shared library; C API, preserved discrete boundary','nodes':len(vertices),'tetrahedra':len(tets),'min_determinant_design_units_cubed':float(det.min()),'nonpositive_tets':int((det<=0).sum()),'source_vertices_max_error':float(d.max()),'boundary_faces_exact':aset==bset,'missing_original_boundary_faces':len(aset-bset),'new_boundary_faces':len(bset-aset),'total_volume':float(det.sum()/6),'scope':'Conforming geometry candidate only, pending tissue-quality/region/contact gates; no solve.'};(O/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));assert r['boundary_faces_exact'] and not r['nonpositive_tets']
