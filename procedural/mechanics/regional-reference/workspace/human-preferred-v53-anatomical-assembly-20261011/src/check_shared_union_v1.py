from pathlib import Path
import numpy as np,json,hashlib,sys
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];O=B/'shared-junction-v1';sys.path.insert(0,str(R/'human-preferred-v53-mechanics-20261011/src'));from triangle_sat import sat_intersect
q=np.load(O/'geometry.npz');v=q['vertices'];f=q['triangles'];tri=v[f];tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);G=np.load(R/'human-preferred-v53-mechanics-20261011/cheek-remesh-v9/geometry.npz');L=np.load(B/'route-layout-v8/parts.npz');M=np.load(B/'mandible-v3/geometry.npz');H=np.load(R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz');bone=np.load(R/'human-preferred-v53-mechanics-20261011/cheek-perioral-insertion-v3/geometry.npz')
fd={tuple(sorted(p)):i for i,p in enumerate(f)};caps={}
for name,verts,faces in [('maxillary_support',G['vertices'],G['root_faces']),('mandible',L['mandibular_depressor_vertices'],L['mandibular_depressor_triangles'][L['mandibular_origin_local_face_ids']])]:
 d,idx=cKDTree(v).query(verts[faces]);assert d.max()<1e-10;ids=np.array([fd[tuple(sorted(p))] for p in idx]);caps[name]=ids
np.savez_compressed(O/'attachment_face_ids.npz',mandibular_origin_face_ids=caps['mandible'],zygomatic_origin_face_ids=caps['maxillary_support'])
pp=np.array(tree.overlap(tree),int).reshape(-1,2);pp=pp[pp[:,0]<pp[:,1]];inc=(f[pp[:,0],:,None]==f[pp[:,1],None,:]).any((1,2));hits=[];ihits=[]
for st in range(0,len(pp),3000):
 p=pp[st:st+3000];ic=inc[st:st+3000];a=tri[p[:,0]];b=tri[p[:,1]]
 if (~ic).any():hh=sat_intersect(a[~ic],b[~ic],eps=1e-10);hits.extend(p[~ic][hh].tolist())
 if ic.any():
  aa=a[ic];bb=b[ic];ca=aa.mean(1);cb=bb.mean(1);aa=ca[:,None]+(1-1e-6)*(aa-ca[:,None]);bb=cb[:,None]+(1-1e-6)*(bb-cb[:,None]);hh=sat_intersect(aa,bb,eps=1e-12);ihits.extend(p[ic][hh].tolist())
report={'geometry_sha256':hashlib.sha256((O/'geometry.npz').read_bytes()).hexdigest(),'nonincident_crossings':len(hits),'trimmed_incident_crossings':len(ihits),'nonincident_examples':hits[:20],'trimmed_incident_examples':ihits[:20],'exact_root_face_counts':{k:len(x) for k,x in caps.items()},'obstacles':{}}
for name,ov,of in [('skin',H['vertices'],H['triangles'][H['object_ids']==0]),('mandible',M['vertices'],M['triangles']),('maxillary_support',bone['bone_vertices'],bone['bone_triangles'])]:
 bt=BVHTree.FromPolygons(ov.tolist(),of.tolist(),all_triangles=True);pairs=np.array(tree.overlap(bt),int).reshape(-1,2);cross=[];excluded=0
 for st in range(0,len(pairs),3000):
  p=pairs[st:st+3000]
  if name in caps:
   allowed=np.isin(p[:,0],caps[name]);excluded+=int(allowed.sum());p=p[~allowed]
  a=tri[p[:,0]];b=ov[of[p[:,1]]]
  if name in caps:c=a.mean(1);a=c[:,None]+(1-1e-6)*(a-c[:,None])
  if len(a):hh=sat_intersect(a,b,eps=1e-10);cross.extend(p[hh].tolist())
 report['obstacles'][name]={'pairs':len(pairs),'excluded_exact_contact_faces':excluded,'crossings':len(cross),'examples':cross[:15]}
report['admitted_shared_surface']=not hits and not ihits and all(r['crossings']==0 for r in report['obstacles'].values());report['scope']='Connected closed surface only. No tetrahedral elements or new equilibrium. Tested attachment-adjacent faces use 1e-6 centroid trim; retained exact bone-contact faces are excluded explicitly.';(O/'surface_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
