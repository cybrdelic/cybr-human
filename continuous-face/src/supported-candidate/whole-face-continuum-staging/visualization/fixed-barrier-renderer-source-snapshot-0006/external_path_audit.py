"""Validate the external continuous CLOSED-owner physical-site path receipt.

This reader checks the declared external proof and its exact input binding. It
does not run contact queries, alter mechanics or imply global surface safety.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from literal_renderer import sha_file
from fixed_policy_reader import POLICY as FIXED_POLICY, POLICY_SHA, AUDITOR_SHA
from renderer_report_binding import STORAGE_AUDITOR_SHA, STORAGE_HELPER_SHA

ARRAY_HASH_CONVENTION='SHA256(str(array.dtype).encode() + str(array.shape).encode() + C-contiguous array.tobytes())'
CONTACT_SOURCE_FIELDS={
    'coarse_physical_site_contact_epigraph.py':'exterior_branch_provider_sha256',
    'exterior_feature_bundles.py':'exterior_branch_source_sha256',
    'coarse_physical_site_contact.py':'base_contact_source_sha256',
    'coarse_rigid_query.py':'rigid_query_source_sha256',
    'compact_penetration_barrier.py':'compact_barrier_source_sha256',
    'coarse_physical_site_contact_rolling.py':'rolling_adapter_source_sha256',
    'rolling_exterior_bounds.py':'rolling_screen_source_sha256'}
NEW_INTERVAL_DT_SCOPE='Newly requested intervals only; prior actual dt is retained per frame'
RECONSTRUCTION_CONTRACT='fixed_0p005mm_interframe_allowance_v1'
RECONSTRUCTION_LIMITS={'physical_endpoint_penetration_cap_mm':.05,
    'interframe_reconstruction_allowance_mm':.005,'certified_interframe_penetration_cap_mm':.055}
RECONSTRUCTION_020_CONTRACT='fixed_0p02mm_interframe_allowance_v1'
RECONSTRUCTION_020_LIMITS={'physical_endpoint_penetration_cap_mm':.05,
    'interframe_reconstruction_allowance_mm':.02,'certified_interframe_penetration_cap_mm':.07}
RECONSTRUCTION_020_PROOF_SOURCES={
    'continuous_closed_site_reconstruction_020.py':'3b3ba0cee5f4f0882fcb1e0d313df171931f610acfe594ce19c1d02106e829bb',
    'audit_reconstruction_020_intervals.py':'6445be81f2dc7d783bcf1eff5ca3d76671a88351603bed3457255e46ccdce88d'}


def reconstruction_policy(evidence,*,proof_schema_required=True):
    contract=evidence.get('reconstruction_contract')
    if contract==RECONSTRUCTION_CONTRACT:
        limits=RECONSTRUCTION_LIMITS
    elif contract==RECONSTRUCTION_020_CONTRACT:
        limits=RECONSTRUCTION_020_LIMITS
        if (proof_schema_required or 'reconstruction_contract_schema' in evidence) and evidence.get('reconstruction_contract_schema')!='fixed_reconstruction_accuracy_v2' or \
                (proof_schema_required or 'trajectory_error_bound_proven' in evidence) and evidence.get('trajectory_error_bound_proven') is not False:
            raise ValueError('The 0.02 mm allowance requires its explicit versioned approximation schema')
        if not proof_schema_required and not isinstance(evidence.get('reconstruction_allowance_scope'),str):
            raise ValueError('Mechanics must explicitly disclose the fixed geometric reconstruction approximation')
    else:raise ValueError('Unknown fixed reconstruction contract')
    if any(evidence.get(key)!=value for key,value in limits.items()):
        raise ValueError('The fixed reconstruction endpoint/allowance/chord limits changed')
    return contract,limits


def bound_array_sha(array):
    array=np.ascontiguousarray(array)
    h=hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def validate_closed_path_audit(path, frames, case, frame_sha256, report_sha256, mechanics_report):
    path=Path(path)
    audit=json.loads(path.read_text())
    if audit.get('schema')!='continuous_closed_site_external_audit_v1' or audit.get('qualified') is not True:
        raise ValueError('Missing qualified external continuous closed-owner site-cap audit')
    if audit.get('case')!=case or audit.get('accepted_npz_sha256')!=frame_sha256 or \
            audit.get('mechanics_report_sha256')!=report_sha256:
        raise ValueError('External continuous path audit does not bind this exact accepted NPZ/report/case')
    if audit.get('array_hash_convention')!=ARRAY_HASH_CONVENTION or \
            audit.get('source_initial_positions_sha256')!=bound_array_sha(frames['source_initial_positions_mm']):
        raise ValueError('External path audit source array/hash convention mismatch')
    if audit.get('audited_snapshots_unchanged_during_audit') is not True or \
            audit.get('contact_sources_unchanged_during_audit') is not True or \
            audit.get('physical_force_law_changed') is not False or audit.get('physical_history_changed') is not False:
        raise ValueError('External audit must preserve and reverify actual physical state/law')
    for key in ('global_surface_collision_certified','unsigned_open_sheet_non_crossing_certified'):
        if audit.get(key) is not False:
            raise ValueError('External closed-site audit must not imply unsupported global/open-sheet certification')
    sources=audit.get('contact_sources_sha256')
    if not isinstance(sources,dict) or not sources or any(not isinstance(h,str) or len(h)!=64 or
            any(c not in '0123456789abcdef' for c in h) for h in sources.values()):
        raise ValueError('Missing exact contact-source identities for external path check')
    provenance=mechanics_report.get('coarse_contact_provenance',{})
    for file,key in CONTACT_SOURCE_FIELDS.items():
        if file not in sources or sources[file]!=provenance.get(key):
            raise ValueError(f'External check uses different actual contact source: {file}')
    for file in ('continuous_closed_site_path.py','volume_guard.py'):
        if file not in sources:
            raise ValueError(f'Missing external checker source identity: {file}')
    reconstruction=('audit_reconstruction_intervals.py' in sources or 'audit_reconstruction_020_intervals.py' in sources or
        audit.get('reconstruction_contract') is not None or mechanics_report.get('reconstruction_contract') is not None)
    if reconstruction:
        contract,limits=reconstruction_policy(audit)
        if reconstruction_policy(mechanics_report,proof_schema_required=False)!=(contract,limits):
            raise ValueError('Independent proof and mechanics report have different reconstruction contracts')
        names=(('audit_reconstruction_020_intervals.py','continuous_closed_site_reconstruction_020.py')
            if contract==RECONSTRUCTION_020_CONTRACT else ('audit_reconstruction_intervals.py','continuous_closed_site_reconstruction.py'))
        if any(name not in sources for name in names):
            raise ValueError('Bounded reconstruction requires its separately identified auditor and proof helper')
        if contract==RECONSTRUCTION_020_CONTRACT:
            qualified_sources=dict(RECONSTRUCTION_020_PROOF_SOURCES)
            auditor=sources.get('audit_reconstruction_020_intervals.py')
            if auditor in (AUDITOR_SHA,STORAGE_AUDITOR_SHA):
                if audit.get('fixed_barrier_policy_validation_source_sha256')!=POLICY_SHA or \
                        audit.get('fixed_barrier_policy_validation_source_unchanged') is not True:
                    raise ValueError('Fixed-policy external auditor lacks its exact unchanged validator source')
                if auditor==STORAGE_AUDITOR_SHA and (audit.get('report_storage_source_sha256')!=STORAGE_HELPER_SHA or
                        audit.get('report_storage_source_unchanged') is not True):
                    raise ValueError('Storage-aware auditor lacks its exact unchanged report decoder')
                qualified_sources['audit_reconstruction_020_intervals.py']=auditor
            if any(sources.get(name)!=digest for name,digest in qualified_sources.items()):
                raise ValueError('The 0.02 mm reconstruction proof does not use a frozen qualified helper/auditor')
            if any(r.get('barrier_policy')==FIXED_POLICY for r in mechanics_report.get('runs',{}).get(case,{}).get('reports',[]) if r.get('accepted') is True) and auditor not in (AUDITOR_SHA,STORAGE_AUDITOR_SHA):
                raise ValueError('Fixed-policy frames require the reviewed fixed-policy external auditor')
    incremental=('audit_new_accepted_intervals.py' in sources or reconstruction)
    if not incremental and 'audit_accepted_site_chords.py' not in sources:
        raise ValueError('Missing known exact external audit runner identity')
    for key in ('source_domain_sha256','source_phase_sha256','source_support_sha256','site_resolution_mm'):
        if key not in audit or audit[key]!=mechanics_report.get(key):
            raise ValueError(f'External check uses a different physical source: {key}')
    query_data=audit.get('physical_query_data_sha256',{})
    for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
        if key not in query_data or query_data[key]!=provenance.get(key):
            raise ValueError(f'External check uses different actual physical query data: {key}')
    if audit.get('saved_velocity_history_consistent') is not True:
        raise ValueError('External check did not verify saved velocity/time history')
    times=frames['physical_times_s']
    if audit.get('accepted_frames_audited')!=len(times) or not np.array_equal(times,np.asarray(audit.get('physical_times_s'))):
        raise ValueError('External path audit times/count differ from accepted states')
    intervals=audit.get('intervals')
    if not isinstance(intervals,list) or len(intervals)!=len(times):
        raise ValueError('Every actual accepted physical interval requires its own external audit')
    dt_scope=mechanics_report.get('physical_dt_s_scope')
    if dt_scope not in (None,NEW_INTERVAL_DT_SCOPE):
        raise ValueError('Unknown physical timestep history semantics')
    prefix=0
    if incremental:
        prefix=audit.get('reused_prefix_intervals')
        digest=audit.get('reused_prefix_audit_sha256')
        if type(prefix) is not int or not 0<prefix<=len(times) or \
                not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Incremental audit needs an exact nonempty qualified prefix binding')
        audited_dt=np.asarray(audit.get('physical_interval_dt_s',[]))
        if audit.get('new_intervals_audited')!=len(times)-prefix or audited_dt.shape!=times.shape or not np.allclose(
                audited_dt,np.diff(np.r_[0.,times]),rtol=1e-12,atol=1e-15):
            raise ValueError('Incremental audit counts or actual physical intervals differ')
        for i,interval in enumerate(intervals):
            if i<prefix:
                if interval.get('reused_without_new_queries') is not True or interval.get('reused_from_audit_sha256')!=digest:
                    raise ValueError('Reused physical interval lacks its exact prior proof hash')
            elif interval.get('reused_without_new_queries') is True:
                raise ValueError('New interval cannot be labeled a reused prefix')
    elif dt_scope is not None:
        raise ValueError('Scoped adaptive history requires the incremental external audit runner')
    accepted_rows=[r for r in mechanics_report.get('runs',{}).get(case,{}).get('reports',[]) if r.get('accepted') is True]
    if len(accepted_rows)!=len(times):
        raise ValueError('External audit needs the corresponding accepted mechanics rows')
    for i,interval in enumerate(intervals):
        q0=frames['source_initial_positions_mm'] if i==0 else frames['positions_mm'][i-1]
        v0=np.zeros_like(q0) if i==0 else frames['velocities_mm_s'][i-1]
        t0=0. if i==0 else float(times[i-1])
        dt=float(times[i])-t0
        if ((dt_scope is None or i>=prefix) and not np.isclose(
                dt,mechanics_report.get('physical_dt_s',float('nan')),rtol=1e-12,atol=1e-15)) or \
                not np.isclose(dt,accepted_rows[i].get('dt_s',float('nan')),rtol=1e-12,atol=1e-15) or \
                not np.allclose(frames['velocities_mm_s'][i],(frames['positions_mm'][i]-q0)/dt,rtol=2e-12,atol=2e-12):
            raise ValueError('Actual saved velocity/report timestep differs from its physical interval')
        if interval.get('qualified') is not True or interval.get('frame_index')!=i or \
                interval.get('t0_s')!=t0 or interval.get('t1_s')!=float(times[i]):
            raise ValueError('Missing or mismatched qualified actual physical interval')
        for key,array in (('q0_sha256',q0),('q1_sha256',frames['positions_mm'][i]),('previous_velocity_sha256',v0)):
            if interval.get(key)!=bound_array_sha(array):
                raise ValueError(f'External interval {i} does not bind actual {key}')
        chord=interval.get('site_chord',{})
        if chord.get('continuous_closed_owner_site_cap_checked') is not True or chord.get('feasible') is False:
            raise ValueError('Actual physical site chord lacks a passed continuous closed-owner cap check')
        cap=chord.get('penetration_cap_mm')
        if cap==.05:
            if chord.get('source_sha256')!=sources['continuous_closed_site_path.py']:
                raise ValueError('Strict 0.05 mm chord proof lacks its exact strict helper identity')
        elif cap==.055 and reconstruction:
            if 'continuous_closed_site_reconstruction.py' not in sources or chord.get('source_sha256')!=sources['continuous_closed_site_reconstruction.py'] or \
                    chord.get('physical_contact_law_changed') is not False or any(
                    chord.get(key)!=value for key,value in RECONSTRUCTION_LIMITS.items()):
                raise ValueError('Bounded chord proof must retain the separate unchanged endpoint cap and fixed geometric allowance')
            if accepted_rows[i].get('reconstruction_contract')!=RECONSTRUCTION_CONTRACT:
                raise ValueError('A reconstructed physical interval needs the same explicit mechanics-frame contract')
        elif cap==.07 and reconstruction and contract==RECONSTRUCTION_020_CONTRACT:
            if reconstruction_policy(chord)!=(contract,limits) or \
                    chord.get('source_sha256')!=sources['continuous_closed_site_reconstruction_020.py'] or \
                    chord.get('physical_contact_law_changed') is not False or \
                    accepted_rows[i].get('reconstruction_contract')!=contract:
                raise ValueError('A 0.07 mm proof needs the new exact helper, versioned frame contract and unchanged physical force law')
        else:
            raise ValueError('Unsupported occupied-site chord cap; never silently loosen the endpoint/contact law')
        if reconstruction:
            endpoint=accepted_rows[i].get('endpoint',{}).get('maximum_site_penetration_mm',float('nan'))
            if not np.isfinite(endpoint) or not 0<=endpoint<=.05:
                raise ValueError('Bounded reconstruction does not relax the unchanged 0.05 mm solved-endpoint cap')
            if accepted_rows[i].get('reconstruction_contract') is not None:
                _,frame_limits=reconstruction_policy(accepted_rows[i],proof_schema_required=False)
                if frame_limits['certified_interframe_penetration_cap_mm']>limits['certified_interframe_penetration_cap_mm']:
                    raise ValueError('Accepted frame uses a looser reconstruction policy than the report/proof')
        if interval.get('positive_J_path',{}).get('certified') is not True:
            raise ValueError('Actual physical interval lacks the external positive-J path certificate')
    return {'audit_file':str(path.resolve()),'audit_sha256':sha_file(path),'external_evidence':audit,
            'bounded_interframe_geometric_allowance_mm':limits['interframe_reconstruction_allowance_mm'] if reconstruction else 0.,
            'geometric_allowance_is_not_a_total_temporal_or_state_error_bound':True,
            'renderer_recomputed_geometry_proof':False,
            'global_surface_collision_certified':False,'unsigned_open_sheet_non_crossing_certified':False}
