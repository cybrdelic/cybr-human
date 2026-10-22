import sys,json,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from coarse_rigid_query import NativeRigidQuery,bounded_band_fixture
r=NativeRigidQuery.from_native();data=json.load(open(P/'coarse-first-direction-offenders.json'))['rows'][0]['offenders'];p=np.array([x['current'] for x in data]);new=r.query_band(p);rows=[];error=0.
for i,x in enumerate(data):
 faces=r.owners[int(x['old_owner'])]['ids'];qq,dd=r._finite_witness(np.broadcast_to(p[i],(len(faces),3)),faces);k=int(np.argmin(dd));depth=float(np.sqrt(dd[k]));inside,valid=r.containment(p[[i]]);assert inside[0,np.flatnonzero(r.owner_ids==int(x['old_owner']))[0]]
 if depth<=.05:
  assert new['finite_valid'][i] and abs(new['signed_gap'][i]+depth)<1e-10
  for j in range(3):
   plus,minus=p[[i]].copy(),p[[i]].copy();plus[0,j]+=1e-6;minus[0,j]-=1e-6
   fd=(r.query_band(plus)['signed_gap'][0]-r.query_band(minus)['signed_gap'][0])/2e-6;error=max(error,abs(fd-new['normal'][i,j]))
 else:assert not new['finite_valid'][i] and new['penetration_limit_exceeded'][i]
 rows.append(dict(site=x['site'],owner=x['old_owner'],old_gap_mm=x['old_gap'],complete_band_gap_mm=float(new['signed_gap'][i]),all_owner_facets_exact_depth_mm=depth,compared_native_facets=len(faces),within_penetration_limit=bool(depth<=.05)))
assert error<1e-7
out=dict(passed=True,scope='Ten exact failed trajectory points, brute all-facets owner check, finite-difference derivative and synthetic finite-feature regression only; no transient or global collision claim',failed_point_count=len(rows),rows=rows,maximum_gradient_error=error,synthetic=bounded_band_fixture())
(P/'coarse-band-repair-qualification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
