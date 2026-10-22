"""One uncommitted physical KKT direction, for shared finite-feature diagnosis."""
import json
import time
from pathlib import Path
import importlib.util
import numpy as np
from scipy.sparse import diags
from source_transfer import load_geometry_with_source_fibres, sha
from clipped_origin_support import load_support_bundle
from implicit_dynamics import ImplicitDynamics
from linear_equilibration import solve_kkt

here = Path(__file__).resolve().parent
out = here/'coarse-barrier-v1/shared-checkpoint-inspection'
started = time.perf_counter()
solid = load_geometry_with_source_fibres(here.parent/'geometry/whole-face-continuum-final.npz',
    here.parent/'geometry/whole-face-muscle-phases-final.npz')
support, _ = load_support_bundle(solid, here/'coarse-physical-site-v1/supports')
module_path = here.parent/'contact/coarse_physical_site_contact.py'
spec = importlib.util.spec_from_file_location('diagnostic_guard', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
guard = module.create_guard(solid, support)
z = np.load(out/'passive-step0001-NONPHYSICAL-checkpoint.npz')
meta = json.loads((out/'passive-step0001-NONPHYSICAL-checkpoint.json').read_text())
q, previous, velocity = (z[k].copy() for k in
    ('nonphysical_iterate_mm', 'q_previous_mm', 'velocity_previous_mm_s'))
guard.set_barrier_scale(meta['barrier_scale'])
dt, activation = meta['dt_s'], meta['activation']
dynamics = ImplicitDynamics(solid, guard, constraints=support, guard_scope='coarse_physical_sites')
e, g, detail = dynamics.evaluate(q, previous, previous+dt*velocity, dt, activation)
Kmat = solid.tangent(q, activation)
K = Kmat + guard.tangent(q) + diags(np.repeat(solid.mass/dt**2, 3))
d, reactions, linear = solve_kkt(K, -g.ravel(), support.C3, equilibrate=True)
d = d.reshape(q.shape)
checks = []
for eps in (1e-2, 1e-3):
    ep, gp, _ = solid.energy(q+eps*d, activation)
    em, gm, _ = solid.energy(q-eps*d, activation)
    e0, gm0, _ = solid.energy(q, activation)
    finite = (gp-gm).ravel()/(2*eps)
    exact = Kmat@d.ravel()
    checks.append(dict(epsilon=eps, tangent_relative_error=float(np.linalg.norm(finite-exact)/max(np.linalg.norm(exact),1e-30)),
        tangent_max_absolute_error_N=float(np.max(np.abs(finite-exact))),
        directional_energy_error_N_mm=float(abs((ep-em)/(2*eps)-np.sum(gm0*d)))))
np.savez_compressed(out/'coupled-direction.npz', positions_mm=q, direction_mm=d,
    q_previous_mm=previous, velocity_previous_mm_s=velocity,
    dynamic_gradient_N=g, source_positions_mm=solid.X)
report = dict(scope='ONE unregularized physical KKT direction; NONPHYSICAL diagnostic, never a frame',
    accepted=False, dt_s=dt, barrier_scale=meta['barrier_scale'], activation=activation,
    implicit_objective_N_mm=e, dynamic_residual_max_N=detail['dynamic_residual_max_N'],
    relative_dynamic_balance_max=detail['relative_dynamic_balance_max'],
    directional_slope_N_mm=float(np.sum(g*d)), linear_system=linear,
    maximum_carrier_direction_mm=float(np.linalg.norm(d, axis=1).max()),
    maximum_occupied_site_direction_mm=guard.max_physical_site_difference(q, q+d),
    material_tangent_checks=checks, guard_source_sha256=sha(module_path),
    npz_sha256=sha(out/'coupled-direction.npz'), wall_seconds=time.perf_counter()-started)
(out/'coupled-direction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2), flush=True)
