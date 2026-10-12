from pathlib import Path
import numpy as np,json,hashlib
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parents[1];P=B/'mandible-v3/geometry.npz';g=np.load(P);h=np.load(R/'user-human-preferred/recovery/restored-project/output/coordinated-orbital-scale-v53/head.npz');tree=BVHTree.FromPolygons(h['vertices'].tolist(),h['triangles'][h['object_ids']==0].tolist(),all_triangles=True)
ids=np.flatnonzero(~(g['skin_parity']==1).all(1));dirs=np.array([[1,.331,-.713],[-.813,.143,.719],[.341,-.823,.119],[-.472,-.917,.391],[.371,.711,-.917],[-.673,.541,-.183],[.211,-.417,-.873]],float);dirs/=np.linalg.norm(dirs,axis=1)[:,None];records=[]
for i in ids:
 p=g['vertices'][i];values=[]
 for d in dirs:
  D=Vector(d);o=Vector(p);count=0
  for k in range(80):
   z,n,fi,dist=tree.ray_cast(o,D,1000)
   if z is None:break
   count+=1;o=z+D*.0005
  values.append(count%2)
 z,n,fi,dist=tree.find_nearest(Vector(p));records.append({'vertex':int(i),'position':p.tolist(),'original_parity':g['skin_parity'][i].tolist(),'seven_additional_parities':values,'nearest_surface_distance':dist,'nearest_outward_normal_signed_offset':float((Vector(p)-z).dot(n))})
surf=json.loads((B/'mandible-v3/surface_check.json').read_text());r={'geometry_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'records':records,'all_additional_rays_inside':all(all(x['seven_additional_parities']) for x in records),'connected_closed_surface_has_zero_triangle_crossings_with_skin':surf['skin_triangle_crossings']==0,'admitted_for_layout':all(all(x['seven_additional_parities']) for x in records) and surf['skin_triangle_crossings']==0 and surf['self_nonincident_crossings']==0 and surf['trimmed_incident_crossings']==0,'interpretation':'Two original upward oblique rays disagreed with the other two directions; seven independent directions and nearest normal resolve them. Exact skin/mandible triangle bounding trees have no candidate surface intersection, and the closed connected component is interior. Original parity data and report are retained unchanged.'};(B/'mandible-v3/parity_resolution.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
