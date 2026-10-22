"""One unchanged-history, unregularized epigraph QP diagnostic; no state commit."""
from pathlib import Path
import json,time,importlib.util
import numpy as np
from scipy.sparse import diags
from source_transfer import load_geometry_with_source_fibres,sha
from clipped_origin_support import load_support_bundle
from implicit_dynamics import ImplicitDynamics
from exterior_epigraph import epigraph_direction
root=Path(__file__).resolve().parent;out=root/'coarse-barrier-v1/reconstruction-sequence-v1/accuracy-inspection';out.mkdir(exist_ok=False)
p=root/'coarse-barrier-v1/reconstruction-sequence-v1/active-solve/zygomaticus_frontalis-step0003-NONPHYSICAL-checkpoint.npz'
meta=json.loads(p.with_suffix('.json').read_text());assert sha(p)==meta['npz_sha256']
started=time.perf_counter();solid=load_geometry_with_source_fibres(root.parent/'geometry/whole-face-continuum-final.npz',root.parent/'geometry/whole-face-muscle-phases-final.npz')
support,_=load_support_bundle(solid,root/'coarse-physical-site-v1/supports')
gpath=root.parent/'contact/coarse_physical_site_contact_epigraph.py';spec=importlib.util.spec_from_file_location('accuracy_guard',gpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);guard=module.create_guard(solid,support)
with np.load(p) as z:q,previous,velocity=[z[k].copy() for k in ('nonphysical_iterate_mm','q_previous_mm','velocity_previous_mm_s')]
guard.set_barrier_scale(meta['barrier_scale']);dt=meta['dt_s'];activation=meta['activation'];prediction=previous+dt*velocity
solver=ImplicitDynamics(solid,guard,constraints=support,guard_scope='coarse_physical_sites')
e,g,detail=solver.evaluate(q,previous,prediction,dt,activation)
K=solid.tangent(q,activation)+guard.tangent(q)+diags(np.repeat(solid.mass/dt**2,3));branches=guard.contact_branches(q)
d,mixed,slope,linear,epigraph=epigraph_direction(K,g.ravel(),support.C3,branches,inexact_search_after_operations=None)
d=d.reshape(q.shape);generalized=solver.generalized_contact_detail(detail,q,g,mixed,epigraph,previous,prediction,dt)
probes=[]
for alpha in (.25,.5,1.):
 et,_,det=solver.evaluate(q+alpha*d,previous,prediction,dt,activation)
 probes.append(dict(alpha=alpha,objective_N_mm=et,objective_difference_N_mm=et-e,feasible=det['feasible']))
np.savez_compressed(out/'unregularized-direction.npz',q_mm=q,direction_mm=d,previous_mm=previous,velocity_mm_s=velocity)
report=dict(accepted=False,scope='One unregularized original physical KKT/epigraph direction, no changed mass/stiffness/damping/pins or committed state',
 checkpoint_sha256=sha(p),dt_s=dt,barrier_scale=meta['barrier_scale'],selected_gradient_slope_N_mm=float(g.ravel()@d.ravel()),
 true_max_branch_directional_slope_N_mm=slope,maximum_carrier_correction_mm=float(np.linalg.norm(d,axis=1).max()),
 maximum_occupied_site_correction_mm=guard.max_physical_site_difference(q,q+d),generalized=generalized,
 linear=linear,epigraph=epigraph,objective_probes=probes,numerical_regularization=0.,wall_seconds=time.perf_counter()-started)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('generalized','linear','epigraph')},indent=2),flush=True)
print('generalized residual',generalized['dynamic_residual_max_N'],'relative',generalized['relative_dynamic_balance_max'],flush=True)
