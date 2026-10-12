from pathlib import Path
import sys,numpy as np,json,hashlib
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'human-preferred-v53-mechanics-20261011/src'));from triangle_sat import sat_intersect
P=B/'route-layout-v8/parts.npz';q=np.load(P);h=np.load(R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz');m=np.load(B/'mandible-v3/geometry.npz');g=np.load(R/'human-preferred-v53-mechanics-20261011/cheek-perioral-insertion-v3/geometry.npz')
obstacles={'skin':(h['vertices'],h['triangles'][h['object_ids']==0]),'mandible':(m['vertices'],m['triangles']),'maxillary_support':(g['bone_vertices'],g['bone_triangles'])};trees={k:BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True) for k,(v,f) in obstacles.items()};names=[k[:-9] for k in q.files if k.endswith('_vertices')];res={}
for name in names:
 v=q[name+'_vertices'];f=q[name+'_triangles'];tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True);tri=v[f];report={}
 e=np.r_[f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]];_,ic,c=np.unique(np.sort(e,1),axis=0,return_inverse=True,return_counts=True);wind=np.bincount(ic,weights=np.where(e[:,0]<e[:,1],1,-1));report['nonmanifold_edges']=int((c!=2).sum());report['inconsistent_winding_edges']=int((wind!=0).sum())
 pp=np.array(tree.overlap(tree),int).reshape(-1,2);pp=pp[pp[:,0]<pp[:,1]];inc=(f[pp[:,0],:,None]==f[pp[:,1],None,:]).any((1,2));hits=[];ihits=[]
 for st in range(0,len(pp),3000):
  a=pp[st:st+3000];ic=inc[st:st+3000];ta=tri[a[:,0]].copy();tb=tri[a[:,1]].copy()
  if (~ic).any():hh=sat_intersect(ta[~ic],tb[~ic],eps=1e-10);hits.extend(a[~ic][hh].tolist())
  if ic.any():
   ta=ta[ic];tb=tb[ic];ca=ta.mean(1);cb=tb.mean(1);ta=ca[:,None]+(1-1e-6)*(ta-ca[:,None]);tb=cb[:,None]+(1-1e-6)*(tb-cb[:,None]);hh=sat_intersect(ta,tb,eps=1e-12);ihits.extend(a[ic][hh].tolist())
 report['self_nonincident_crossings']=len(hits);report['self_trimmed_incident_crossings']=len(ihits);report['self_examples']=hits[:10];report['self_trimmed_examples']=ihits[:10]
 for key,(ov,of) in obstacles.items():
  pairs=np.array(tree.overlap(trees[key]),int).reshape(-1,2);cross=[];excluded=0
  for st in range(0,len(pairs),3000):
   p=pairs[st:st+3000];a=tri[p[:,0]].copy();b=ov[of[p[:,1]]]
   if name=='mandibular_depressor' and key=='mandible':
    # The literal origin cap is the declared bone contact. Other faces are tested
    # after a tiny trim of their shared attachment edge, not spatially excluded.
    allowed=np.isin(p[:,0],q['mandibular_origin_local_face_ids']);excluded+=int(allowed.sum());p=p[~allowed];a=a[~allowed];b=b[~allowed];ca=a.mean(1);a=ca[:,None]+(1-1e-6)*(a-ca[:,None])
   if name=='zygomatic_reference' and key=='maxillary_support':continue # exact earlier distributed contact proof, unchanged
   if len(a):hh=sat_intersect(a,b,eps=1e-10);cross.extend(p[hh].tolist())
  report[key]={'candidate_pairs':len(pairs),'crossings':len(cross),'excluded_declared_cap_pairs':excluded,'examples':cross[:12]}
 res[name]=report;print(name,json.dumps(report),flush=True)
(B/'route-layout-v8/surface_checks.json').write_text(json.dumps({'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'parts':res,'limits':['Existing unchanged zygomatic contact reuses its earlier distributed-root proof.','Declared new mandibular root faces excluded from obstacle crossing counts, their exact coordinate correspondence is separately pinned.','Shared adjacent attachment edges use1e-6 centroid trim; no exact claim in that strip.','Intentional inter-part junction overlaps remain pending conformal union; these are not independent disjoint tissues.']},indent=2))
