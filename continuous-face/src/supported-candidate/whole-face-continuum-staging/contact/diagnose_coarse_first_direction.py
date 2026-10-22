import sys,time,json
from pathlib import Path
import numpy as np
from scipy.sparse import diags
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P.parent/'mechanics')]
from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle
from coarse_physical_site_contact import create_guard
from implicit_dynamics import ImplicitDynamics
from linear_equilibration import solve_kkt
start=time.perf_counter();s=load_geometry_with_source_fibres(P.parent/'geometry/whole-face-continuum-final.npz',P.parent/'geometry/whole-face-muscle-phases-final.npz');c,_=load_support_bundle(s,P.parent/'mechanics/coarse-physical-site-v1/supports');g=create_guard(s,c);solver=ImplicitDynamics(s,g,constraints=c,guard_scope='coarse_physical_sites');q=s.X.copy();dt=.01
e,grad,detail=solver.evaluate(q,q,q,dt,{})
K=s.tangent(q,{})+g.tangent(q)+diags(np.repeat(s.mass/dt**2,3))
d,_,rep=solve_kkt(K,-grad.ravel(),c.C3,equilibrate=True);d=d.reshape(q.shape);slope=float(np.sum(grad*d));alpha=min(solver.safe_volume_step_bound(q,d),g.safe_step_bound(q,d))
rows=[]
for a in [alpha]:
 trial=q+a*d;ee,_,r=solver.evaluate(trial,q,q,dt,{})
 row=dict(alpha=a,endpoint=g.endpoint(trial),objective=ee,initial_objective=e,armijo_limit=e+1e-4*a*slope,armijo_satisfied=bool(ee<=e+1e-4*a*slope));s0=g._state(trial);bad=np.argsort(s0['gap'])[:10];site=s0['ids'][bad];old=g.rigid.query(s0['points'][site]);new=g.rigid.query_band(s0['points'][site],clearance=g.h,penetration_limit=g.penetration_limit);ref=g.rigid.query(g.P0[site]);row['offenders']=[dict(site=int(site[i]),reference=g.P0[site[i]].tolist(),current=s0['points'][site[i]].tolist(),motion=float(s0['motion'][site[i]]),old_gap=float(old['signed_gap'][i]),new_gap=float(new['signed_gap'][i]),reference_gap=float(ref['signed_gap'][i]),old_owner=int(old['owner_id'][i]),new_owner=int(new['owner_id'][i]),reference_owner=int(ref['owner_id'][i])) for i in range(len(site))];rows.append(row);print(json.dumps(row),flush=True)
report=dict(scope='Source first linear direction and three rejected/accepted trial probes only; no state accepted or saved',seconds=time.perf_counter()-start,rows=rows,linear_system=rep)
(P/'coarse-first-direction-offenders.json').write_text(json.dumps(report,indent=2)+'\n')
