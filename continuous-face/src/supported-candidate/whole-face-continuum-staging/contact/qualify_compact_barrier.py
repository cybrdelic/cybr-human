"""Quantify the tested interior-point law without running actual-face dynamics."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from compact_penetration_barrier import compact_barrier_terms
from test_compact_penetration_barrier import fixture
P=Path(__file__).resolve().parent
solid,guard=fixture();q=solid.X.copy();d=np.random.default_rng(11).normal(size=q.shape);eps=1e-7
e,g=guard.energy(q);fd=(guard.energy(q+eps*d)[0]-guard.energy(q-eps*d)[0])/(2*eps);gradient_error=abs(fd-float(np.sum(g*d)))
H=guard.tangent(q);fdg=(guard.energy(q+eps*d)[1]-guard.energy(q-eps*d)[1])/(2*eps);tangent_error=float(np.max(abs(fdg.ravel()-H@d.ravel())))
account=guard.contact_accounting(q,include_vectors=False);loaded=[];quadratic=[]
for scale in [1.,.1,.01]:
 mu=1.5e-5*scale;F=.0015;s=mu/(F+mu/.01);b=compact_barrier_terms([s-.05],mu)
 loaded.append(dict(scale=scale,mu_N_mm=mu,slack_mm=s,reaction_N=float(b['reaction'][0]),force_balance_error_N=abs(float(b['reaction'][0])-F),lambda_s_N_mm=float(b['complementarity'][0])))
 residual=lambda slack:.01*(.1+.05-slack)+compact_barrier_terms([slack-.05],mu)['reaction'][0]-.01
 s=brentq(residual,1e-10,.01,xtol=1e-14);b=compact_barrier_terms([s-.05],mu)
 quadratic.append(dict(scale=scale,slack_mm=s,quadratic_force_N=.01*(.1+.05-s),constraint_reaction_N=float(b['reaction'][0]),load_N=.01,balance_error_N=abs(residual(s)),lambda_s_N_mm=float(b['complementarity'][0])))
direction=np.tile([0.,0.,-1.],(4,1));alpha=guard.safe_step_bound(q,direction)
report=dict(passed=True,scope='Analytic stop and bounded synthetic contact qualification only, not a whole-face transient',tests=dict(compact_barrier=7,existing_coarse_adapter=4),energy_gradient_directional_error=gradient_error,planar_fixture_normal_tangent_error=tangent_error,normal_tangent_min_eigenvalue=float(np.linalg.eigvalsh(H.toarray()).min()),normal_tangent_is_exact_general_distance_hessian=False,analytic_loaded_stop=loaded,loaded_stop_with_unchanged_quadratic=quadratic,multi_owner_accounting=account,fraction_to_boundary_alpha=alpha,post_fraction_endpoint_feasible=guard.endpoint(q+alpha*direction)['feasible'],crossing_endpoint_rejected=not guard.endpoint(q+.006*direction)['feasible'],crossing_path_rejected=not guard.path(q,q+.006*direction)['feasible'],unchanged_physical_parameters=dict(penalty_N_per_mm4=.01,clearance_mm=.1,penetration_cap_mm=.05,carrier_J_floor=.1,trial_increment_mm=.5,path_spacing_mm=.25),numerical_barrier_band_mm=.01,source_sha256={x:hashlib.sha256((P/x).read_bytes()).hexdigest() for x in ['compact_penetration_barrier.py','coarse_physical_site_contact.py','coarse_rigid_query.py','test_compact_penetration_barrier.py']})
assert gradient_error<1e-9 and tangent_error<1e-7
(P/'compact-barrier-qualification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
