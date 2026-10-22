"""Finite h-band geometry/sign data for raw SEARCH predictors only.

No barrier, force, tangent, energy, physical cache, or validity override. The
complete finite witness verification below matches the existing curvature
verifier, without its reaction/stiffness assembly. Native invalid flags remain
invalid. Every physically invalid queried site is checked, not just anchors.
"""
from collections import Counter
import copy
import time
import numpy as np

POLICY = 'cap_only_raw_predictor_h_band_geometry_v1'


def cap_only_raw_endpoint(detail):
    endpoint = detail.get('endpoint', {})
    slack = endpoint.get('minimum_penetration_slack_mm', np.inf)
    return (detail.get('feasible') is False
        and detail.get('reason') == 'finite endpoint guard'
        and np.isfinite(slack) and slack <= 0)


def _geometry_rows(guard, native, state):
    from interior_feature_curvature import _feature
    # Geometry state is assembled only from a separately valid native query.
    selected = np.flatnonzero(state['active'] & (state['force_gap'] < 0))
    sites = state['force_ids'][selected]; owners = state['force_owner'][selected]
    distances = -state['force_gap'][selected]
    points = state['points'][sites]
    scale = np.max(abs(native.vertices))
    pads = 2048*np.finfo(float).eps*(np.linalg.norm(points, axis=1)+scale+guard.h+1)
    candidates = [[] for _ in selected]
    for start in range(0, len(selected), 256):
        x = points[start:start+256]; distance = distances[start:start+256]; pad = pads[start:start+256]
        for face_ids, tree, radius in native.distance_bins:
            choices = tree.query_ball_point(x, radius+distance+pad)
            count = np.fromiter(map(len, choices), int, len(x))
            if not count.sum():
                continue
            pi = np.repeat(np.arange(len(x)), count)
            fi = face_ids[np.concatenate(choices).astype(int)]
            keep = native.face_owner[fi] == owners[start+pi]
            pi, fi = pi[keep], fi[keep]
            for begin in range(0, len(pi), 16384):
                ii, jj = pi[begin:begin+16384], fi[begin:begin+16384]
                tri = native.triangles[jj]
                delta = np.maximum(np.maximum(tri.min(1)-x[ii], x[ii]-tri.max(1)), 0.)
                keep = np.sum(delta**2, axis=1) <= (distance[ii]+pad[ii])**2
                ii, jj = ii[keep], jj[keep]
                if not len(ii):
                    continue
                witness, d2 = native._finite_witness(x[ii], jj)
                dd = np.sqrt(d2)
                keep = dd <= distance[ii]+pad[ii]
                for i, f, w, d in zip(ii[keep], jj[keep], witness[keep], dd[keep]):
                    candidates[start+i].append((float(d), int(f), w))
    rows = []; skipped = Counter(); face_count = 0
    for group, force_row in enumerate(selected):
        if not native.owners[int(owners[group])]['closed']:
            skipped['not_closed_owner'] += 1; continue
        p, d, pad = points[group], distances[group], pads[group]
        if d <= pad:
            skipped['zero_or_unresolved_distance'] += 1; continue
        choices = sorted(candidates[group], key=lambda x:(x[0], x[1]))
        if not choices:
            skipped['missing_complete_witness'] += 1; continue
        dd, face, witness = choices[0]
        normal = state['normal'][force_row]
        if dd != d or not np.array_equal(-(p-witness)/d, normal):
            skipped['winner_not_exactly_reproduced'] += 1; continue
        tied = [c for c in choices if c[0] <= d+pad]
        if any(np.linalg.norm(c[2]-witness) > pad for c in tied):
            skipped['ambiguous_different_closest_points'] += 1; continue
        features = []; reason = None
        for _, f, w in tied:
            feature, reason = _feature(native, p, f, w, normal, pad)
            if reason:
                break
            features.append(feature)
        if reason:
            skipped[reason] += 1; continue
        if len({x['key'] for x in features}) != 1:
            skipped['ambiguous_different_features'] += 1; continue
        feature = features[0]
        if feature['kind'] == 'face':
            face_count += 1; continue
        rows.append(dict(force_row=int(force_row), site=int(sites[group]), owner=int(owners[group]),
            gap_mm=float(-d), closest_point_mm=witness.copy(), normal=normal.copy(),
            tied_faces=np.array([c[1] for c in tied], int), **feature))
    return dict(rows=rows, report=dict(interior_owner_terms=len(selected),
        stable_edge_terms=sum(r['kind']=='edge' for r in rows),
        stable_vertex_terms=sum(r['kind']=='vertex' for r in rows),
        stable_face_terms_zero_addition=face_count,
        skipped_ambiguous_or_unresolved=dict(skipped),
        complete_finite_witness_enumeration=True, geometry_only=True))


def raw_predictor_geometry(guard, plain, anchor):
    """Check cached physical failures at h/h on a separate native query object.

    The cached failed physical state is READ ONLY. This routine never calls
    _state, interior_curvature_data, compact_barrier_terms, or any energy/force
    method. It never supplies a physical state, only optional geometric rows.
    """
    started = time.perf_counter()
    info = dict(policy=POLICY, geometry_only=True, usable_as_physical_state=False,
        raw_barrier_evaluated=False, physical_cache_written=False, qualified=False)
    raw = np.asarray(plain, float)
    state = getattr(guard, 'cache', None)
    if (state is None or not np.isfinite(raw).all()
            or not np.array_equal(state.get('q'), raw)):
        return dict(rows=[], report=dict(info, reason='Matching finite cached raw query required'))
    valid = np.asarray(state['valid'], bool); ids = np.asarray(state['ids'], int)
    bad = ids[~valid]
    # Physical native invalid sites must actually exhibit a cap exceedance.
    # The new query still has to establish their finite geometry and sign.
    if np.any(np.asarray(state['gap'])[~valid] >= -guard.penetration_limit):
        return dict(rows=[], report=dict(info, reason='Raw invalid site is not a cap exceedance'))
    anchor_sites = np.array(sorted({key[0] for key in anchor}), int)
    cap_ids = state['force_ids'][state['force_gap']+guard.penetration_limit <= 0]
    query_sites = np.unique(np.r_[anchor_sites, bad, cap_ids])
    if not len(query_sites) or not np.isfinite(guard.h) or guard.h < guard.penetration_limit:
        return dict(rows=[], report=dict(info, reason='Finite diagnostic band and nonempty sites required'))
    native = getattr(guard.rigid, 'native', guard.rigid)
    context = getattr(guard, '_soc_geometry_query_context', None)
    if context is None or context['physical_native'] is not native or context['band_mm'] != guard.h:
        diagnostic = copy.copy(native)
        diagnostic.report = copy.deepcopy(native.report)
        context = dict(physical_native=native, band_mm=guard.h, query=diagnostic)
        guard._soc_geometry_query_context = context
    diagnostic = context['query']
    points = np.asarray(guard.C[query_sites]@raw)
    if not np.isfinite(points).all():
        return dict(rows=[], report=dict(info, reason='Nonfinite raw sites'))
    result = diagnostic.query_band(points, clearance=guard.h,
        penetration_limit=guard.h, owner_terms=True)
    info.update(diagnostic_clearance_mm=float(guard.h), diagnostic_penetration_limit_mm=float(guard.h),
        physical_penetration_limit_mm=float(guard.penetration_limit), queried_sites=len(query_sites),
        invalid_physical_sites_rechecked=len(bad), query_scope='current anchor union all cached raw invalid and nonpositive-slack sites')
    required = ('signed_gap', 'normal', 'contact_signed_gap', 'contact_normal')
    if (not np.asarray(result['valid'], bool).all()
            or not all(np.isfinite(result[k]).all() for k in required)
            or np.any(np.asarray(result['signed_gap']) < -guard.h)
            or np.any(np.asarray(result['contact_signed_gap']) < -guard.h)):
        return dict(rows=[], report=dict(info, reason='Diagnostic native geometry/sign invalid or deeper than finite band'))
    # Verify that all old invalid sites really became valid cap-only sites.
    bad_local = np.searchsorted(query_sites, bad)
    if np.any(np.asarray(result['signed_gap'])[bad_local] >= -guard.penetration_limit):
        return dict(rows=[], report=dict(info, reason='Diagnostic does not reproduce raw cap exceedance'))
    force_local = np.asarray(result['contact_site_index'], int)
    global_sites = query_sites[force_local]
    keep = np.array([(int(i),int(o)) in anchor for i,o in zip(global_sites,result['contact_owner_id'])], dtype=bool)
    expected = {(int(i),int(o)) for i,o in zip(global_sites[keep],result['contact_owner_id'][keep])}
    if expected != set(anchor):
        return dict(rows=[], report=dict(info, reason='Diagnostic missing current anchor owner'))
    # Local indices avoid allocating all physical sites for a small query.
    geometry = dict(points=points, force_ids=force_local[keep],
        force_owner=np.asarray(result['contact_owner_id'])[keep],
        force_gap=np.asarray(result['contact_signed_gap'])[keep],
        normal=np.asarray(result['contact_normal'])[keep], active=np.ones(int(keep.sum()),bool))
    data = _geometry_rows(guard, diagnostic, geometry)
    for row in data['rows']:
        row['site'] = int(query_sites[row['site']])
    data['report'].update(info, qualified=True, wall_seconds=time.perf_counter()-started)
    return data
