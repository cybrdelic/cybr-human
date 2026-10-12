from pathlib import Path
import numpy as np,json,hashlib
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];O=B/'route-layout-v8';P=O/'parts.npz';q=np.load(P);h=np.load(R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz');tree=BVHTree.FromPolygons(h['vertices'].tolist(),h['triangles'][h['object_ids']==0].tolist(),all_triangles=True)
dirs=np.array([[1,.331,-.713],[-.813,.143,.719],[.341,-.823,.119],[-.472,-.917,.391],[.371,.711,-.917],[-.673,.541,-.183],[.211,-.417,-.873]],float);dirs/=np.linalg.norm(dirs,axis=1)[:,None]
def parity(p,ds):
 out=[]
 for d in ds:
  D=Vector(d);o=Vector(p);count=0
  for k in range(80):
   z,n,fi,dist=tree.ray_cast(o,D,1000)
   if z is None:break
   count+=1;o=z+D*.0005
  out.append(count%2)
 return out
records=[]
for name in ['zygomatic_reference','junction','superficial_oral','deep_oral','buccinator_deep','mandibular_depressor']:
 v=q[name+'_vertices'];old=q[name+'_parity'];ids=np.flatnonzero(~(old==1).all(1));
 if name=='mandibular_depressor':
  pp=np.array([parity(p,dirs[:3]) for p in v]);ids=np.flatnonzero(~(pp==1).all(1));np.save(O/'mandibular_final_parity.npy',pp)
 for i in ids:
  z,n,fi,dist=tree.find_nearest(Vector(v[i]));records.append({'part':name,'vertex':int(i),'seven_additional_parities':parity(v[i],dirs),'distance':dist,'nearest_normal_offset':float((Vector(v[i])-z).dot(n))})
checks=json.load(open(O/'surface_checks.json'));good=all(all(r['seven_additional_parities']) for r in records)
for name,r in checks['parts'].items():
 good=good and all(r[k]==0 for k in ['nonmanifold_edges','inconsistent_winding_edges','self_nonincident_crossings','self_trimmed_incident_crossings']) and all(r[k]['crossings']==0 for k in ['skin','mandible','maxillary_support'])
rep=json.load(open(O/'report.json'));release={'parts_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'admitted_for_geometry_layout':bool(good),'parity_resolution':records,'mandibular_barycentric_reconstruction_exact':bool(np.array_equal(q['mandibular_origin_positions'],q['mandibular_depressor_vertices'][q['mandibular_origin_local_vertex_ids']])),'bone_edge_roundoff_repair':rep['bone_edge_roundoff_repair'],'region_names':['zygomatic_reference','junction','superficial_oral','deep_oral','buccinator_deep','mandibular_depressor'],'limitations':['Individual closed regions overlap intentionally at junction. No conforming common tissue mesh exists yet.','Only mandibular depressor and original zygomatic reference have explicit bone footprints. Other cut ends are unbound.','No new active/passive finite elements, dynamics, TMJ, teeth, calibrated fibre architecture or full oral ring.','Existing earlier cheek equilibrium is not transferred onto this new assembly.','All coordinates and thicknesses are authored estimates, guided by cited layer relations.']};(O/'RELEASE.json').write_text(json.dumps(release,indent=2));print(json.dumps(release,indent=2));assert good
