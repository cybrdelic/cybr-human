"""Fail-closed reader of the real mechanics/first_step.py accepted-frame schema.

This module neither solves nor fabricates frames. It requires matching model and
initial-state identities, solver convergence and non-fixture geometric evidence.
An explicitly separate coarse physical-site mode does not claim a global
render-surface collision certificate.
The renderer validates the evidence supplied by mechanics; it is not a new proof.
"""
import json
from pathlib import Path
import numpy as np
from literal_renderer import sha_file, sha_array, reconstruct_boundary
from report_storage import read_report
from renderer_report_binding import validate_report_storage_receipt
from external_path_audit import validate_closed_path_audit,RECONSTRUCTION_020_CONTRACT
from paired_cursor_binding import validate_paired_cursor_binding
from stationarity_accuracy import validate_stationarity_policy, validate_stationarity_accuracy
from tangent_source_identity import validate_epigraph_factory_identity
from fixed_policy_reader import POLICY as FIXED_POLICY, validate_fixed_sources, validate_fixed_row, sensitivity_summary

EPIGRAPH_STATIONARITY='finite_exterior_energy_epigraph_v1'
EPIGRAPH_BRANCHES='complete_exterior_finite_energy_branches_v1'
WINNER_STATIONARITY='exact_selected_winner_already_stationary'


def validate_generalized_stationarity(row):
    """Check supplied finite-feature evidence, not a seam-projected surrogate."""
    if row.get('contact_generalized_stationarity_qualified') is not True:
        raise ValueError('Missing qualified finite-feature generalized stationarity')
    evidence=row.get('generalized_contact')
    if not isinstance(evidence,dict):
        raise ValueError('Missing finite-feature epigraph evidence')
    kind=evidence.get('kind')
    if kind not in (EPIGRAPH_STATIONARITY,WINNER_STATIONARITY) or row.get('exterior_contact_stationarity')!=kind:
        raise ValueError('Unknown or inconsistent exterior stationarity contract')
    for key in ('energy_law_unchanged','actual_comaximality_satisfied','branch_weights_nonnegative'):
        if evidence.get(key) is not True:
            raise ValueError(f'Missing generalized reaction qualification: {key}')
    if evidence.get('seam_rows_are_source_support_pins') is not False:
        raise ValueError('A seam-projected residual is not accepted generalized stationarity')
    projection=row.get('residual_projection_scope',evidence.get('residual_projection_scope'))
    if projection!='source_origin_support_reactions_only':
        raise ValueError('Only source-origin support reactions may be removed from the residual')
    tolerance=evidence.get('comaximality_tolerance_N_mm',float('nan'))
    if tolerance!=1e-12:
        raise ValueError('Changed finite-feature co-maximality tolerance')
    values={key:evidence.get(key,float('nan')) for key in
        ('group_weight_sum_error','maximum_used_branch_energy_gap_N_mm',
         'max_branch_complementarity_N_mm','sum_branch_complementarity_N_mm')}
    if any(not np.isfinite(v) or v<0 for v in values.values()):
        raise ValueError('Nonfinite or negative epigraph weight/complementarity accounting')
    if values['group_weight_sum_error']>1e-10:
        raise ValueError('Finite-feature weights do not form valid group convex combinations')
    if values['maximum_used_branch_energy_gap_N_mm']>tolerance or values['max_branch_complementarity_N_mm']>tolerance:
        raise ValueError('An off-seam branch mixture is not accepted convergence')
    groups=evidence.get('mixed_groups')
    if not isinstance(groups,list):
        raise ValueError('Missing finite-feature branch mixture accounting')
    if kind==WINNER_STATIONARITY:
        if groups or any(values.values()):
            raise ValueError('Exact selected-winner special case must have zero mixture gaps/complementarity')
    else:
        selected=row.get('smooth_selected_branch_dynamic_residual_max_N',float('nan'))
        if not np.isfinite(selected) or selected<0:
            raise ValueError('Smooth selected-branch residual must remain separately disclosed')
    group_ids=set()
    for group in groups:
        if group.get('group') in group_ids:
            raise ValueError('Duplicate finite-feature mixture group')
        group_ids.add(group.get('group'))
        weights=np.asarray(group.get('weights',[]),float)
        gaps=np.asarray(group.get('branch_energy_gaps_N_mm',[]),float)
        if weights.ndim!=1 or len(weights)<2 or gaps.shape!=weights.shape or \
                not np.isfinite(weights).all() or not np.isfinite(gaps).all() or \
                np.any(weights<0) or np.any(gaps<0) or np.any(gaps>tolerance) or abs(weights.sum()-1)>1e-10:
            raise ValueError('Invalid or off-seam finite-feature branch mixture')
        for key in ('branch_indices','branch_site','branch_owner','branch_triangle'):
            ids=np.asarray(group.get(key,[]))
            if ids.shape!=weights.shape or not np.issubdtype(ids.dtype,np.integer):
                raise ValueError(f'Missing branch/source witness identity: {key}')
        if len(set(group['branch_indices']))!=len(weights):
            raise ValueError('Duplicate branch in a finite-feature mixture')


def load_accepted_frames(path, report_path, case, geometry, closed_path_audit_path=None,paired_cursor_snapshot=None):
    path,report_path=Path(path),Path(report_path)
    stored_report=read_report(report_path)
    report=stored_report.document
    validate_stationarity_policy(report)
    contact_mode=report.get('contact_mode','finite_surface')
    contact_enforcement=report.get('contact_enforcement')
    if contact_enforcement not in (None,'compact_cap_barrier_v1'):
        raise ValueError('Unknown contact-enforcement acceptance contract')
    barrier=contact_enforcement=='compact_cap_barrier_v1'
    provenance=report.get('coarse_contact_provenance',{})
    exterior_stationarity=report.get('exterior_contact_stationarity')
    epigraph=(exterior_stationarity is not None or provenance.get('exterior_branch_representation') is not None)
    tangent_identity=None
    if epigraph:
        if closed_path_audit_path is None:
            raise ValueError('Epigraph state awaits its external continuous closed-owner physical-path audit')
        if not barrier or exterior_stationarity!=EPIGRAPH_STATIONARITY or \
                provenance.get('exterior_branch_representation')!=EPIGRAPH_BRANCHES or \
                provenance.get('original_minimum_distance_energy_unchanged') is not True:
            raise ValueError('Unrecognized finite-feature epigraph model contract')
        hashes=[report.get('epigraph_mechanics_source_sha256'),report.get('guard_module_sha256'),
                provenance.get('exterior_branch_provider_sha256'),provenance.get('exterior_branch_source_sha256')]
        if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError('Missing exact epigraph provider/helper source hashes')
        tangent_identity=validate_epigraph_factory_identity(report,paired_cursor_snapshot)
    if contact_mode not in ('finite_surface','coarse_physical_sites'):
        raise ValueError('Unknown contact approximation mode')
    coarse=contact_mode=='coarse_physical_sites'
    if coarse:
        resolution=report.get('site_resolution_mm',float('nan'))
        if report.get('global_surface_collision_certified') is not False or not np.isfinite(resolution) or resolution<=0:
            raise ValueError('Coarse contact requires explicit resolution and no global-surface certificate claim')
    if barrier and not coarse:
        raise ValueError('The compact barrier contract is for coarse physical sites, not a surface proof')
    if report.get('source_domain_sha256') != geometry['sha256']:
        raise ValueError('Accepted run geometry hash mismatch, including active-node subset')
    if case not in report.get('runs',{}): raise ValueError('Case not present in mechanics report')
    run=report['runs'][case]
    actual_file_sha=sha_file(path)
    if run.get('accepted_npz_sha256') != actual_file_sha:
        raise ValueError('Saved accepted-frame NPZ hash is missing or differs from the mechanics case report')
    with np.load(path,allow_pickle=False) as z:
        required=('positions_mm','velocities_mm_s','physical_times_s','source_initial_positions_mm','scale_mm_per_L0')
        if any(k not in z.files for k in required): raise ValueError('Incomplete accepted-frame file')
        frames={k:z[k] for k in required}
    q=frames['positions_mm']; v=frames['velocities_mm_s']; ts=frames['physical_times_s']
    scale=float(frames['scale_mm_per_L0']); q0=frames['source_initial_positions_mm']
    if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid physical unit scale')
    if not np.array_equal(q0,geometry['nodes']*scale):
        raise ValueError('Actual saved initial q does not exactly match geometry node ordering and units')
    n=len(q)
    if q.ndim!=3 or q.shape[1:]!=q0.shape or v.shape!=q.shape or ts.shape!=(n,) or n==0:
        raise ValueError('No genuine frames or invalid dimensions')
    if any(not np.isfinite(x).all() for x in (q,v,ts,q0)): raise ValueError('Non-finite physical frame')
    if np.any(ts<=0) or np.any(np.diff(ts)<=0): raise ValueError('Physical time must be positive and strictly increasing')
    if run.get('accepted_frames')!=n or not np.array_equal(ts,np.asarray(run.get('actual_times_s'))):
        raise ValueError('Saved times/count do not match mechanics acceptance report')
    rows=[r for r in run.get('reports',[]) if r.get('accepted') is True]
    if len(rows)!=n: raise ValueError('Missing per-frame acceptance evidence')
    fixed_identity=None
    if any("barrier_policy" in r for r in rows):
        if not barrier or not epigraph or any(r.get("barrier_policy")!=FIXED_POLICY for r in rows if "barrier_policy" in r):
            raise ValueError("Unknown or unsupported per-frame barrier policy")
        fixed_identity=validate_fixed_sources(report,paired_cursor_snapshot)
    stationarity_routes=[]
    for i,r in enumerate(rows):
        if r.get('contact_enforcement') != contact_enforcement:
            raise ValueError('Frame contact-enforcement contract mismatch')
        if epigraph:
            validate_generalized_stationarity(r)
        elif r.get('generalized_contact') is not None or r.get('exterior_contact_stationarity') is not None:
            raise ValueError('Per-frame epigraph evidence lacks its model-level identity contract')
        if r.get('status')!='converged' or r.get('fixture_only') is not False:
            raise ValueError('Frame is unconverged or lacks explicit non-fixture qualification')
        if r.get('physical_time_s') != float(ts[i]): raise ValueError('Per-frame physical time mismatch')
        path_evidence=r.get('frame_path',{})
        if coarse:
            if path_evidence.get('feasible') is not True or path_evidence.get('coarse_whole_segment_checked') is not True:
                raise ValueError('Missing declared coarse physical-site step check')
        elif path_evidence.get('feasible') is not True or path_evidence.get('whole_segment_certified') is not True:
            raise ValueError('Missing accepted whole-physical-step geometric certificate')
        if r.get('feasible') is not True: raise ValueError('Missing finite feasible endpoint evidence')
        if r.get('endpoint',{}).get('feasible') is not True:
            raise ValueError('Missing explicit geometric endpoint evidence')
        minimum_J=r.get('minimum_J',float('nan'))
        if not np.isfinite(minimum_J) or minimum_J<=0:
            raise ValueError('Missing positive-J endpoint evidence')
        stationarity_routes.append(validate_stationarity_accuracy(r,report))
        if barrier and 'barrier_policy' in r:
            validate_fixed_row(r,report,q0 if i==0 else q[i-1],
                np.zeros_like(q0) if i==0 else v[i-1],q[i],0. if i==0 else float(ts[i-1]),
                case,validate_generalized_stationarity,validate_stationarity_accuracy)
        elif barrier:
            if r.get('contact_enforcement') != contact_enforcement:
                raise ValueError('Frame contact-enforcement contract mismatch')
            for key in ('barrier_continuation_qualified', 'lower_mu_comparison_passed',
                        'final_inner_stationarity_converged','final_physical_path_qualified',
                        'strict_positive_slack','complementarity_bound_satisfied',
                        'dynamic_residual_includes_barrier_reaction'):
                if r.get(key) is not True:
                    raise ValueError(f'Missing compact barrier acceptance evidence: {key}')
            if r.get('physical_frame_path_deferred') is not False or \
                    r.get('acceptance_scope')!='physical_frame_with_final_parameter_source_to_endpoint_path':
                raise ValueError('An internal mu-stage endpoint is not an accepted physical frame')
            difference=r.get('lower_mu_site_difference_mm',float('nan'))
            site_tolerance=r.get('lower_mu_site_tolerance_mm',float('nan'))
            if not np.isfinite(difference) or not np.isfinite(site_tolerance) or \
                    not 0<site_tolerance<=.005 or not 0<=difference<site_tolerance:
                raise ValueError('Lower-mu occupied-site comparison fails its agreed 0.005 mm budget')
            accounting=r.get('contact_accounting')
            if not isinstance(accounting,dict) or not accounting:
                raise ValueError('Missing approximate-complementarity contact accounting')
            if accounting.get('complementarity_bound_satisfied') is not True:
                raise ValueError('Declared approximate-complementarity bound is not satisfied')
            for key in ('barrier_scale','penalty_energy_N_mm','barrier_energy_N_mm',
                        'max_lambda_s_N_mm','sum_lambda_s_N_mm','maximum_active_mu_N_mm'):
                value=accounting.get(key,float('nan'))
                if not np.isfinite(value) or value<0:
                    raise ValueError(f'Missing finite nonnegative contact accounting: {key}')
            if accounting['barrier_scale']<=0 or not isinstance(accounting.get('per_owner'),list):
                raise ValueError('Missing active barrier scale or per-owner reaction accounting')
            continuation=r.get('barrier_continuation')
            if not isinstance(continuation,dict) or not continuation:
                raise ValueError('Missing one-physical-step barrier continuation evidence')
            for key in ('qualified','same_physical_time_step','physical_history_unchanged_during_stages'):
                if continuation.get(key) is not True:
                    raise ValueError(f'Missing one-physical-step continuation evidence: {key}')
            if continuation.get('committed_physical_time_steps') != 1:
                raise ValueError('Barrier continuation must commit exactly one physical time step')
            stages=continuation.get('stages',[])
            if len(stages)<2 or any(s.get('not_committed_physical_frame') is not True or
                                  s.get('inner_stationarity_converged') is not True or
                                  s.get('physical_frame_path_deferred') is not True or
                                  s.get('acceptance_scope')!='internal_same_timestep_stationarity_only'
                                  for s in stages):
                raise ValueError('Missing converged, uncommitted inner-stage evidence')
            if epigraph:
                for stage in stages:
                    validate_generalized_stationarity(stage)
                    if stage.get('stationarity_accuracy') is not None or stage.get('approximate_stationarity') is True:
                        validate_stationarity_accuracy(stage,report)
    # Identical consecutive coordinates are not intrinsically invalid physics,
    # but may not be exported as motion: never fabricate a looped-still movie.
    repeated=[i for i in range(1,n) if np.array_equal(q[i-1],q[i])]
    report_sha=stored_report.file_sha256
    external=(None if closed_path_audit_path is None else
              validate_closed_path_audit(closed_path_audit_path,frames,case,actual_file_sha,report_sha,report))
    if external is not None:
        validate_report_storage_receipt(external['external_evidence'],stored_report)
    new_policy=report.get('reconstruction_contract')==RECONSTRUCTION_020_CONTRACT
    if new_policy and (external is None or paired_cursor_snapshot is None):
        raise ValueError('The 0.02 mm reconstruction contract requires a fresh independent proof and admitted paired cursor')
    if paired_cursor_snapshot is not None and external is None:
        raise ValueError('Paired cursor admission must bind an independent physical path proof')
    cursor_binding=(None if paired_cursor_snapshot is None else validate_paired_cursor_binding(
        paired_cursor_snapshot,case,closed_path_audit_path,external['audit_sha256'],ts))
    audit={'frame_file':str(path.resolve()),'frame_sha256':actual_file_sha,
        'frame_validation_source_sha256':{name:sha_file(Path(__file__).parent/name) for name in (
            'accepted_frames.py','external_path_audit.py','stationarity_accuracy.py','paired_cursor_binding.py',
            'fixed_policy_reader.py','fixed_barrier_policy.py','report_storage.py','renderer_report_binding.py')},
        'report_file':str(report_path.resolve()),'report_sha256':report_sha,
        'report_raw_sha256':stored_report.raw_sha256,'report_storage':stored_report.identity(),
        'geometry_sha256':geometry['sha256'],'q0_sha256':sha_array(q0),
        'physical_times_s':ts.tolist(),'accepted_frames':n,'case':case,
        'consecutive_identical_frame_indices':repeated,
        'source_phase_sha256':report.get('source_phase_sha256'),
        'source_model_report_status':report.get('status'),
        'contact_mode':contact_mode,
        'contact_enforcement':contact_enforcement,
        'exterior_contact_stationarity':exterior_stationarity,
        'coarse_contact_provenance':provenance,
        'epigraph_mechanics_source_sha256':report.get('epigraph_mechanics_source_sha256'),
        'external_closed_site_path_audit':external,
        'admitted_paired_cursor':cursor_binding,
        'qualified_numerical_tangent_identity':tangent_identity,
        'paired_numerical_parameter_contrast':report.get('paired_numerical_parameter_contrast') or (
            None if cursor_binding is None else cursor_binding.get('paired_numerical_parameter_contrast')),
        'barrier_scales':report.get('barrier_scales'),
        'reconstruction_contract':report.get('reconstruction_contract'),
        'reconstruction_contract_schema':report.get('reconstruction_contract_schema') or (
            None if external is None else external['external_evidence'].get('reconstruction_contract_schema')),
        'trajectory_error_bound_proven':report.get('trajectory_error_bound_proven',
            None if external is None else external['external_evidence'].get('trajectory_error_bound_proven')),
        'physical_endpoint_penetration_cap_mm':report.get('physical_endpoint_penetration_cap_mm'),
        'interframe_reconstruction_allowance_mm':report.get('interframe_reconstruction_allowance_mm'),
        'certified_interframe_penetration_cap_mm':report.get('certified_interframe_penetration_cap_mm'),
        'reconstruction_allowance_scope':report.get('reconstruction_allowance_scope'),
        'reconstruction_policy_transition':report.get('reconstruction_policy_transition'),
        'stationarity_accuracy_policy':report.get('stationarity_accuracy_policy'),
        'stationarity_accuracy_scope':report.get('stationarity_accuracy_scope'),
        'force_state_dynamics_source_sha256':report.get('force_state_dynamics_source_sha256'),
        'force_state_accuracy_source_sha256':report.get('force_state_accuracy_source_sha256'),
        'validated_stationarity_routes':stationarity_routes,
        'fixed_barrier_policy_identity':fixed_identity,
        'allow_cap_only_gap_search_enabled':report.get('allow_cap_only_gap_search_enabled'),
        'raw_predictor_geometry_source_sha256':report.get('raw_predictor_geometry_source_sha256'),
        'barrier_sensitivity_scope':sensitivity_summary(rows),
        'complementarity_is_exact':False if barrier else None,
        'compact_barrier_acceptance_contract': {
            'reaction_inclusive_stationarity':True,
            'lower_mu_occupied_site_tolerance_mm':.005,
            'lower_mu_tolerance_applies_only_to_performed_comparisons':True,
            'every_frame_has_fresh_lower_mu_comparison':not bool(fixed_identity),
            'no_extra_renderer_complementarity_threshold':True} if barrier else None,
        'accepted_frame_evidence': [{key:r[key] for key in (
            'physical_time_s','dt_s','activation','status','feasible','minimum_J','dynamic_residual_max_N','force_tolerance_N',
            'endpoint','frame_path','contact_enforcement','barrier_continuation_qualified',
            'barrier_policy','fixed_barrier_qualified','barrier_qualification',
            'lower_mu_comparison_passed','lower_mu_site_difference_mm','lower_mu_site_tolerance_mm',
            'final_inner_stationarity_converged','contact_accounting','barrier_continuation',
            'physical_frame_path_deferred','final_physical_path_qualified','acceptance_scope',
            'strict_positive_slack','complementarity_bound_satisfied','dynamic_residual_includes_barrier_reaction',
            'contact_generalized_stationarity_qualified','generalized_contact','exterior_contact_stationarity',
            'residual_projection_scope','smooth_selected_branch_dynamic_residual_max_N',
            'reconstruction_contract','physical_endpoint_penetration_cap_mm',
            'reconstruction_contract_schema','trajectory_error_bound_proven',
            'interframe_reconstruction_allowance_mm','certified_interframe_penetration_cap_mm',
            'reconstruction_allowance_scope','stationarity_accuracy','approximate_stationarity',
            'absolute_force_target_passed','relative_dynamic_balance_max','force_component_magnitude_sum_max_N') if key in r}
            for r in rows],
        'global_surface_collision_certified':False if coarse else report.get('global_surface_collision_certified'),
        'site_resolution_mm':report.get('site_resolution_mm') if coarse else None,
        'renderer_does_not_recertify_physics':True}
    return frames,audit

def boundary_frame(frames,index,geometry):
    return reconstruct_boundary(frames['positions_mm'][index],geometry['tets'],
        geometry['exterior_tet'],geometry['exterior_barycentric'])
