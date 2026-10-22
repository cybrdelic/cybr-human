"""Saved-state comparison only. No dynamics, live source edits or pose output."""
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

HERE = Path(__file__).resolve().parent


def assemble(C, ids, r):
    sites = ids[r['contact_site_index']]
    owners = r['contact_owner_id']
    gaps = r['contact_signed_gap']
    normal = r['contact_normal']
    b = compact_barrier_terms(gaps, np.full(len(gaps), 1.5e-5))
    depth = np.maximum(.1-gaps, 0.)
    gradient = np.asarray(C[sites].T@(-(.01*depth+b['reaction'])[:, None]*normal))
    return dict(sites=sites, owners=owners, gaps=gaps, normal=normal,
                energy=float(np.sum(.005*depth**2+b['energy'])), gradient=gradient,
                valid=bool(r['valid'].all()),
                penetration=float(np.maximum(-r['signed_gap'], 0).max(initial=0)))


def main():
    started = time.perf_counter()
    g = np.load(HERE.parent/'geometry/whole-face-continuum-final.npz')
    X, T = g['nodes'], g['tets']
    inv = np.linalg.inv(np.transpose(X[T[:, 1:]]-X[T[:, 0, None]], (0, 2, 1)))
    solid = SimpleNamespace(X=X, T=T, inv=inv)
    c = np.load(HERE.parent/'geometry/whole-face-integration-cells.npz')
    P0 = c['grid_origin']+(c['cell_ijk']+.5)*float(c['grid_spacing'])
    C, _ = occupied_site_map(solid, P0)
    ref = np.load(HERE/'coarse-site-reference-band-v1.npz')['lower_bound_mm']
    rigid = NativeRigidQuery.from_native()
    screen = RollingExteriorBounds(rigid, P0, ref)
    cases = []
    for version in ('v2', 'v3'):
        path = HERE.parent/f'mechanics/coarse-barrier-v1/first-step-{version}/passive-step0001-NONPHYSICAL-checkpoint.npz'
        data = path.read_bytes()
        with np.load(path) as z:
            q = z['nonphysical_iterate_mm'].copy()
        cases.append((version, q, dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())))
    v3 = cases[-1][1]
    cases.append(('v3_nearby_feasible_translation', v3+[.00003, -.00002, .00001], {}))
    cases.append(('v3_rejected_translation', v3+[.003, -.002, .001], {}))
    rows = []
    for name, q, source in cases:
        P = C@q
        baseline_ids = np.flatnonzero(ref-np.linalg.norm(P-P0, axis=1) <= .1)
        t = time.perf_counter()
        br = rigid.query_band(P[baseline_ids], owner_terms=True)
        bt = time.perf_counter()-t
        baseline = assemble(C, baseline_ids, br)
        t = time.perf_counter()
        accelerated_ids = np.flatnonzero(screen.lower_bound(P) <= .1)
        screen_time = time.perf_counter()-t
        t = time.perf_counter()
        ar = rigid.query_band(P[accelerated_ids], owner_terms=True)
        at = time.perf_counter()-t
        accelerated = assemble(C, accelerated_ids, ar)
        identical = all(np.array_equal(baseline[k], accelerated[k])
                        for k in ('sites', 'owners', 'gaps', 'normal', 'gradient'))
        identical &= baseline['energy'] == accelerated['energy']
        assert identical, name
        assert baseline['valid'] == accelerated['valid']
        assert baseline['penetration'] == accelerated['penetration']
        row = dict(case=name, source=source, baseline_sites=len(baseline_ids),
                   accelerated_sites=len(accelerated_ids),
                   baseline_query_seconds=bt, accelerated_query_seconds=at,
                   screen_seconds=screen_time, active_owner_terms=len(baseline['sites']),
                   energy_N_mm=baseline['energy'] if np.isfinite(baseline['energy']) else None,
                   energy_is_finite=bool(np.isfinite(baseline['energy'])),
                   exact_owner_terms_energy_gradient_match=bool(identical),
                   valid=baseline['valid'], maximum_penetration_mm=baseline['penetration'])
        row['refresh'] = screen.refresh(P, accelerated_ids, ar)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(passed=True, scope='Isolated saved nonphysical-state screen regression; not adopted, no dynamics',
                  unchanged_physical_clearance_mm=.1, unchanged_penetration_limit_mm=.05,
                  numerical_refresh_radius_mm=.5, native_geometry_token=screen.token,
                  cases=rows, total_seconds=time.perf_counter()-started)
    (HERE/'rolling-exterior-bounds-qualification.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
