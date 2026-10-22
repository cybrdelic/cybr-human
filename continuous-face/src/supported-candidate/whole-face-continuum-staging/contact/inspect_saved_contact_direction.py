"""One saved coupled direction: native contact derivatives, no nonlinear solve."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import numpy as np
from coarse_physical_site_contact import occupied_site_map
from coarse_rigid_query import NativeRigidQuery
from compact_penetration_barrier import compact_barrier_terms
from rolling_exterior_bounds import RollingExteriorBounds
from qualify_rolling_exterior_bounds import assemble

HERE = Path(__file__).resolve().parent
source = HERE.parent/'mechanics/coarse-barrier-v1/shared-checkpoint-inspection/coupled-direction.npz'
packet = np.load(source)
q, direction = packet['positions_mm'], packet['direction_mm']
g = np.load(HERE.parent/'geometry/whole-face-continuum-final.npz')
X, T = g['nodes'], g['tets']
solid = SimpleNamespace(X=X, T=T, inv=np.linalg.inv(np.transpose(X[T[:, 1:]]-X[T[:, 0, None]], (0, 2, 1))))
cells = np.load(HERE.parent/'geometry/whole-face-integration-cells.npz')
P0 = cells['grid_origin']+(cells['cell_ijk']+.5)*float(cells['grid_spacing'])
C, _ = occupied_site_map(solid, P0)
P, D = C@q, C@direction
reference_lower = np.load(HERE/'coarse-site-reference-band-v1.npz')['lower_bound_mm']
rigid = NativeRigidQuery.from_native()
screen = RollingExteriorBounds(rigid, P0, reference_lower)
ids = np.flatnonzero(reference_lower-np.linalg.norm(P-P0, axis=1) <= .1)
raw = rigid.query_band(P[ids], owner_terms=True)
baseline = assemble(C, ids, raw)
refresh = screen.refresh(P, ids, raw)


def evaluate(alpha):
    points = P+alpha*D
    selected = np.flatnonzero(screen.lower_bound(points) <= .1)
    result = rigid.query_band(points[selected], owner_terms=True)
    a = assemble(C, selected, result)
    a.update(keys=a['sites']*1000+a['owners'], query_sites=len(selected),
             directional_slope=float(np.sum(a['gradient']*direction)))
    return a


base = evaluate(0.)
assert all(np.array_equal(base[k], baseline[k]) for k in ('sites', 'owners', 'gaps', 'normal', 'gradient'))
assert base['energy'] == baseline['energy']
rows, states = [], {0.: base}
for alpha in (1., .5, .25, .125, .0625, .03125, .015625, .001, -.001, .0001, -.0001, .00001, -.00001):
    s = evaluate(alpha)
    states[alpha] = s
    common, i0, it = np.intersect1d(base['keys'], s['keys'], return_indices=True)
    changes = np.linalg.norm(base['normal'][i0]-s['normal'][it], axis=1)
    biggest = np.argsort(changes)[-4:][::-1]
    row = dict(alpha=alpha, energy_N_mm=s['energy'], energy_change_N_mm=s['energy']-base['energy'],
               directional_slope_N_mm=s['directional_slope'], query_sites=s['query_sites'],
               active_owner_terms=len(s['sites']), valid=s['valid'], maximum_penetration_mm=s['penetration'],
               changed_term_count=len(base['keys'])+len(s['keys'])-2*len(common),
               largest_normal_changes=[dict(site=int(base['sites'][i0[j]]), owner=int(base['owners'][i0[j]]),
                   normal_change=float(changes[j]), reference_point_mm=P0[base['sites'][i0[j]]].tolist(),
                   current_point_mm=P[base['sites'][i0[j]]].tolist(),
                   point_direction_mm=D[base['sites'][i0[j]]].tolist(),
                   base_gap=float(base['gaps'][i0[j]]), trial_gap=float(s['gaps'][it[j]]),
                   base_normal=base['normal'][i0[j]].tolist(), trial_normal=s['normal'][it[j]].tolist())
                   for j in biggest])
    rows.append(row)
    print(json.dumps(row), flush=True)
fd = []
for eps in (.001, .0001, .00001):
    a, b = states[eps], states[-eps]
    value = (a['energy']-b['energy'])/(2*eps)
    fd.append(dict(epsilon=eps, finite_difference_N_mm=value, analytic_N_mm=base['directional_slope'],
                   absolute_error_N_mm=abs(value-base['directional_slope'])))
report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              scope='One saved actual coupled direction, no solver or live source edits',
              baseline_rolling_bit_identical=True, baseline_energy_N_mm=base['energy'],
              baseline_directional_slope_N_mm=base['directional_slope'], refresh=refresh,
              trials=rows, central_derivatives=fd)
(HERE/'saved-contact-direction-inspection.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(central_derivatives=fd)), flush=True)
