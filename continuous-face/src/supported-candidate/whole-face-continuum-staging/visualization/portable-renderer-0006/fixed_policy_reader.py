"""Renderer-only bindings for the reviewed fixed-mu approximation.

This does not solve, certify geometry, or invent sensitivity for untested frames.
The exact reviewed pure policy validator is vendored without changes.
"""
import json
from pathlib import Path
import numpy as np
from literal_renderer import sha_file
from renderer_report_binding import STORAGE_AUDITOR_SHA, STORAGE_HELPER_SHA
from fixed_barrier_policy import POLICY, MODEL, validate_fixed_frame, validate_plan, plan_sha

POLICY_SHA='c0b904d7b0f32649cfd80fd2b82859a26dc3068c1823482de4531d1191fd52a0'
WRAPPER_SHA='7d0da93db207442edb521cba269cf84dd6c5cb6c4d1664fbc97becac4a8ee5f7'
AUDITOR_SHA='1c5f028f5205429fe887dfb6a66ea2304839e77fd38937f841525ac50731372d'
FIXED_SOURCES={
    'mechanics/fixed_barrier_policy.py':POLICY_SHA,
    'mechanics/barrier_reconstruction_continuation.py':WRAPPER_SHA,
    'mechanics/coarse_reconstruction_020_curvature_driver.py':'4b2fb916f7f042e42a6350ce1e9635f72fab0d575c4e7e9644c9a41cd0d06f13',
    'mechanics/paired_reconstruction_020_curvature_sequence.py':'a6fca8ee3596ded16ec48dd31b1dec634f653fb24a45a83d1b0dafcf4c2f1ba3',
    'contact/reconstruction-020-v1/audit_reconstruction_020_intervals.py':AUDITOR_SHA}


DRIVER='mechanics/coarse_reconstruction_020_curvature_driver.py'
CONTROLLER='mechanics/paired_reconstruction_020_curvature_sequence.py'
RAW_HELPER='mechanics/raw_predictor_geometry.py'
RAW_HELPER_SHA='55ef445f5d9982f69dfc8ac621b9a7a3b841134a974b79d4f906c1652831d66d'
QUALIFIED_ENTRYPOINT_PAIRS={
    (FIXED_SOURCES[DRIVER],FIXED_SOURCES[CONTROLLER]):'fixed_policy_v6_through_v9',
    ('60d90718424d1e5e3acdd8e1238a8b0e6d88ad7b4ba7e432843e743ab6c5427a',
     '73c4b211fde52cff0fa60874e869d10ed9157b704cc8ac1521b2207e6c8ec847'):'cap_only_geometry_search_opt_in_v10',
    ('60d90718424d1e5e3acdd8e1238a8b0e6d88ad7b4ba7e432843e743ab6c5427a',
     'dabad12cb8ccd14f93bae7d499619bcf044963a04832c36b0ae0249026cff2bb'):'cap_only_geometry_search_retry_seed_v11',
    ('60d90718424d1e5e3acdd8e1238a8b0e6d88ad7b4ba7e432843e743ab6c5427a',
     '728fc856b3b61ab22b3209440a9ec04919fcb636dd3fa26991c9bf1cb55d7b28'):'cap_only_geometry_search_600s_runtime_v14'}


def validate_fixed_sources(report, cursor):
    if report.get('barrier_policy')!=POLICY or report.get('fixed_barrier_model')!=MODEL:
        raise ValueError('Missing exact fixed barrier report policy/model')
    if report.get('fixed_barrier_policy_source_sha256')!=POLICY_SHA or \
            report.get('barrier_qualification_source_sha256')!=WRAPPER_SHA or \
            sha_file(Path(__file__).with_name('fixed_barrier_policy.py'))!=POLICY_SHA:
        raise ValueError('Fixed barrier qualification source differs from reviewed source')
    if not isinstance(cursor,dict):
        raise ValueError('Fixed barrier frames require the admitted paired cursor')
    sources=cursor.get('document',{}).get('source_sha256',{})
    bindings={}
    def matching(suffix):
        matches=[(p,h) for p,h in sources.items() if p==suffix or p.endswith('/'+suffix)]
        if len(matches)!=1:
            raise ValueError('Paired controller does not uniquely bind source: '+suffix)
        return matches[0]
    pair=tuple(matching(suffix)[1] for suffix in (DRIVER,CONTROLLER))
    revision=QUALIFIED_ENTRYPOINT_PAIRS.get(pair)
    if revision is None:
        raise ValueError('Unknown or mixed fixed-policy driver/controller source pair')
    expected=dict(FIXED_SOURCES);expected.update({DRIVER:pair[0],CONTROLLER:pair[1]})
    auditor_suffix='contact/reconstruction-020-v1/audit_reconstruction_020_intervals.py'
    storage_revision='legacy_plain_report_snapshots'
    if revision=='cap_only_geometry_search_600s_runtime_v14' and matching(auditor_suffix)[1]!=STORAGE_AUDITOR_SHA:
        raise ValueError('Reviewed v14 runtime-only bundle retains the exact storage-aware auditor')
    if matching(auditor_suffix)[1]==STORAGE_AUDITOR_SHA:
        if revision not in ('cap_only_geometry_search_retry_seed_v11','cap_only_geometry_search_600s_runtime_v14'):
            raise ValueError('Storage-aware auditor requires the exact reviewed v12 entrypoint bundle')
        expected[auditor_suffix]=STORAGE_AUDITOR_SHA
        expected['mechanics/report_storage.py']=STORAGE_HELPER_SHA
        storage_revision='lossless_json_gzip_v1_v12'

    if revision in ('cap_only_geometry_search_opt_in_v10','cap_only_geometry_search_retry_seed_v11','cap_only_geometry_search_600s_runtime_v14'):
        expected[RAW_HELPER]=RAW_HELPER_SHA
    for suffix,digest in expected.items():
        path,actual=matching(suffix)
        if actual!=digest:
            raise ValueError('Paired controller does not bind reviewed fixed source: '+suffix)
        bindings[path]=digest
    if 'allow_cap_only_gap_search_enabled' in report:
        if type(report['allow_cap_only_gap_search_enabled']) is not bool or revision not in ('cap_only_geometry_search_opt_in_v10','cap_only_geometry_search_retry_seed_v11','cap_only_geometry_search_600s_runtime_v14') or \
                report.get('raw_predictor_geometry_source_sha256')!=RAW_HELPER_SHA:
            raise ValueError('Search-only opt-in/source provenance differs from reviewed v10')
    plan=validate_plan(report.get('fixed_barrier_check_plan',{}))
    if report.get('fixed_barrier_check_plan_sha256')!=plan_sha(plan):
        raise ValueError('Fixed barrier report plan hash differs')
    plan_path=Path(report['fixed_barrier_check_plan_path'])
    digest=report.get('fixed_barrier_check_plan_file_sha256')
    if sha_file(plan_path)!=digest or json.loads(plan_path.read_bytes())!=plan:
        raise ValueError('Original report check-plan bytes differ from their immutable identity')
    # A restarted controller stores an exact-byte copy at its own output path.
    # Preserve both original and current bindings; path aliasing is not a change
    # of plan, and semantic equality alone is deliberately insufficient.
    direct=sources.get(str(plan_path))==digest
    candidates=[Path(p) for p,h in sources.items()
        if h==digest and Path(p).name=='fixed-barrier-check-plan.json']
    if direct:
        current_plan_path=plan_path
    elif len(candidates)==1:
        current_plan_path=candidates[0]
        if current_plan_path.read_bytes()!=plan_path.read_bytes():
            raise ValueError('Controller plan alias is not the exact original report plan bytes')
    else:
        raise ValueError('Fixed barrier plan is not bound uniquely by the admitted controller')
    if sha_file(current_plan_path)!=digest or json.loads(current_plan_path.read_bytes())!=plan:
        raise ValueError('Controller-bound immutable plan has changed')
    controller=cursor['document']
    if (controller.get('fixed_barrier_check_plan') is not None and controller['fixed_barrier_check_plan']!=plan) or \
            (controller.get('fixed_barrier_check_plan_sha256') is not None and controller['fixed_barrier_check_plan_sha256']!=plan_sha(plan)):
        raise ValueError('Controller plan metadata differs from exact bound plan bytes')
    bindings[str(current_plan_path)]=digest
    plan_bindings={str(plan_path):digest,str(current_plan_path):digest}
    return {'policy':POLICY,'controller_entrypoint_source_revision':revision,
        'controller_source_manifest_bindings':bindings,'report_storage_source_revision':storage_revision,
        'immutable_plan_file_bindings':plan_bindings,
        'original_report_plan_path':str(plan_path),'controller_plan_path':str(current_plan_path),
        'original_and_controller_plan_file_sha256':digest,
        'controller_plan_path_is_exact_byte_alias':str(plan_path)!=str(current_plan_path),
        'plan_sha256':plan_sha(plan),'ordinary_frames_have_fresh_sensitivity_check':False,
        'empirical_checks_are_not_total_trajectory_error_bounds':True}


def validate_accounting(accounting):
    if not isinstance(accounting,dict) or accounting.get('complementarity_bound_satisfied') is not True:
        raise ValueError('Missing qualified fixed-stage contact accounting')
    for key in ('barrier_scale','penalty_energy_N_mm','barrier_energy_N_mm',
                'max_lambda_s_N_mm','sum_lambda_s_N_mm','maximum_active_mu_N_mm'):
        value=accounting.get(key,float('nan'))
        if not np.isfinite(value) or value<0:
            raise ValueError('Missing finite nonnegative fixed-stage accounting: '+key)
    if accounting['barrier_scale']<=0 or not isinstance(accounting.get('per_owner'),list):
        raise ValueError('Missing fixed-stage per-owner accounting')
    def finite(value):
        if isinstance(value,dict): return all(finite(v) for v in value.values())
        if isinstance(value,list): return all(finite(v) for v in value)
        return not isinstance(value,(float,int)) or np.isfinite(value)
    if not finite(accounting):
        raise ValueError('Nonfinite fixed-stage contact accounting')


def validate_fixed_row(row, report, previous, velocity, candidate, previous_time,
                       case, generalized_validator, accuracy_validator):
    qualification=validate_fixed_frame(row,previous,velocity,candidate)
    if 'barrier_continuation' in row:
        raise ValueError('Fixed frame must not fabricate legacy continuation evidence')
    request=qualification['check_request']
    if request['previous_physical_time_s']!=previous_time:
        raise ValueError('Fixed frame original physical time differs')
    # Prefixes from earlier immutable plans may remain, but cannot claim checks
    # outside the current immutable plan or silently lose a prior target.
    plan=report['fixed_barrier_check_plan']
    if any(check not in plan['checks'] for check in request['plan']['checks']):
        raise ValueError('Fixed frame plan differs from the retained report plan')
    if request['plan']!=plan:
        future_old=[c for c in request['plan']['checks'] if c['target_time_s']<=row['physical_time_s']]
        future_new=[c for c in plan['checks'] if c['target_time_s']<=row['physical_time_s']]
        if future_old!=future_new:
            raise ValueError('A new plan cannot retroactively omit a due empirical check')
    checked=qualification['comparison_performed']
    if qualification.get('comparison_time_s')!=(row['physical_time_s'] if checked else None) or \
            qualification.get('comparison_required') is not checked or \
            row.get('lower_mu_site_tolerance_mm')!=.005:
        raise ValueError('Fixed frame empirical-check scope is inconsistent')
    ramp=report.get('activation_ramp_s',float('nan'))
    if not np.isfinite(ramp) or ramp<=0 or case not in ('passive','zygomaticus_frontalis'):
        raise ValueError('Unknown fixed-frame activation history')
    expected={} if case=='passive' else dict.fromkeys(('zygomaticus','frontalis'),.3*float(np.clip(row['physical_time_s']/ramp,0,1)))
    if qualification.get('activation')!=expected or row.get('activation')!=expected:
        raise ValueError('Fixed frame activation differs from actual prescribed time/case')
    stages=qualification['stages']
    for stage in stages:
        if stage.get('activation')!=expected or stage.get('dt_s')!=row['dt_s']:
            raise ValueError('Fixed stage lost original case activation or physical interval')
        generalized_validator(stage)
        accuracy_validator(stage,report)
        validate_accounting(stage.get('contact_accounting'))
        if stage['contact_accounting']['barrier_scale']!=stage['barrier_scale']:
            raise ValueError('Fixed stage accounting belongs to a different mu')
    selected=stages[qualification['selected_stage_index']]
    for key in ('generalized_contact','exterior_contact_stationarity','residual_projection_scope',
                'dynamic_residual_max_N','force_tolerance_N','stationarity_accuracy',
                'approximate_stationarity','absolute_force_target_passed',
                'relative_dynamic_balance_max','force_component_magnitude_sum_max_N',
                'support','support_constraint_error_mm','support_reaction_max_N',
                'dominant_dynamic_residual','dynamic_residual_L2_N'):
        if row.get(key)!=selected.get(key):
            raise ValueError('Accepted frame is not the selected UPPER stationarity/reaction state: '+key)
    validate_accounting(row.get('contact_accounting'))
    for key in ('quadratic_contact_force_net_N','quadratic_contact_force_representation','finite_exterior_epigraph'):
        if row['contact_accounting'].get(key)!=selected['contact_accounting'].get(key):
            raise ValueError('Accepted accounting lost selected UPPER reactions: '+key)
    return qualification


def sensitivity_summary(rows):
    fixed=[(i,r) for i,r in enumerate(rows) if r.get('barrier_policy')==POLICY]
    return {'policy':POLICY if fixed else None,
        'fixed_frame_indices':[i for i,r in fixed],
        'fixed_empirically_checked_frame_indices':[i for i,r in fixed if r['barrier_qualification']['comparison_performed']],
        'fixed_empirically_checked_physical_times_s':[r['physical_time_s'] for i,r in fixed if r['barrier_qualification']['comparison_performed']],
        'fixed_ordinary_frame_indices_without_fresh_comparison':[i for i,r in fixed if not r['barrier_qualification']['comparison_performed']],
        'legacy_per_frame_continuation_indices':[i for i,r in enumerate(rows) if 'barrier_policy' not in r],
        'no_interpolation_of_sensitivity_indicators':True,'no_total_trajectory_error_bound_claim':True}


def recheck_fixed_plan_bindings(audit):
    identity=audit.get('fixed_barrier_policy_identity')
    if identity is None:
        return
    for path,digest in identity['immutable_plan_file_bindings'].items():
        if sha_file(Path(path))!=digest:
            raise RuntimeError('Immutable original/controller check plan changed during rendering')
