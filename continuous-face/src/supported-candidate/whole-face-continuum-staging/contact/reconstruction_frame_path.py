"""Final saved-frame reconstruction check only; never a Newton trial guard.

Endpoints retain their physical 0.05 mm cap and complete endpoint validity.
The affine inter-frame reconstruction has a separately declared 0.005 mm
geometric allowance, not a temporal/state-error estimate. Force laws, internal
Newton path checks, and all saved coordinates/history remain unchanged.
"""
from pathlib import Path
import numpy as np
from coarse_physical_site_contact import sha
from continuous_closed_site_reconstruction import certify_closed_owner_reconstruction_chord
from volume_guard import certify_positive_volume


def reconstruction_frame_path(guard,q0,q1):
    q0,q1=np.asarray(q0,float),np.asarray(q1,float)
    if q0.shape!=q1.shape or q0.shape!=guard.solid.X.shape or not np.isfinite(q0).all() or not np.isfinite(q1).all():
        raise ValueError('Finite shape-matched physical endpoints required')
    if guard.penetration_limit!=.05 or not np.isfinite(guard.path_spacing) or guard.path_spacing<=0:
        raise ValueError('Unchanged physical endpoint cap and finite positive resolution required')
    result=dict(feasible=False,coarse_whole_segment_checked=False,
        scope='Final inter-frame occupied-site reconstruction, not a Newton search-path guard',
        reconstruction_contract='fixed_0p005mm_interframe_allowance_v1',
        physical_endpoint_penetration_cap_mm=.05,interframe_reconstruction_allowance_mm=.005,
        certified_interframe_penetration_cap_mm=.055,physical_contact_law_changed=False,
        global_surface_collision_certified=False,unsigned_open_sheet_non_crossing_certified=False,
        source_sha256=sha(Path(__file__)),
        continuous_reconstruction_source_sha256=sha(Path(__file__).with_name('continuous_closed_site_reconstruction.py')),
        volume_guard_source_sha256=sha(Path(__file__).with_name('volume_guard.py')))
    volume=certify_positive_volume(guard.solid,q0,q1,floor=guard.jacobian_floor)
    result['volume']=volume
    if not volume['certified']:return dict(result,reason='positive-J path')
    increment=float(np.linalg.norm(guard.C@(q1-q0),axis=1).max(initial=0))
    if not np.isfinite(increment):return dict(result,reason='Nonfinite physical-site chord')
    count=max(1,int(np.ceil(increment/guard.path_spacing)))
    result.update(maximum_site_increment_mm=increment,legacy_resolution_cells=count,
        maximum_resolution_cell_length_mm=increment/count,legacy_maximum_resolution_cells=128)
    if count>128:return dict(result,reason='Original bounded path-resolution budget exceeded')
    endpoints=[guard.endpoint(q0),guard.endpoint(q1)];result['endpoint_checks']=endpoints
    if not all(x.get('feasible') is True for x in endpoints):
        return dict(result,reason='Unchanged physical endpoint guard failed')
    result['maximum_endpoint_penetration_mm']=max(x['maximum_site_penetration_mm'] for x in endpoints)
    chord=certify_closed_owner_reconstruction_chord(guard,q0,q1);result['site_chord']=chord
    good=bool(chord['feasible'] and chord['continuous_closed_owner_site_cap_checked'])
    return dict(result,feasible=good,coarse_whole_segment_checked=good,reason=chord['reason'])
