"""Validate the declared local approximate-stationarity alternative explicitly.

This is not a new force solve and is not a bound on total state/trajectory error.
Generalized reaction validity remains independently checked by accepted_frames.
"""
import numpy as np

POLICY='absolute_force_or_full_qp_force_state_accuracy_v1'


def validate_stationarity_policy(report,*,require_sources=False):
    policy=report.get('stationarity_accuracy_policy')
    if policy is None:
        return
    if policy!=POLICY:
        raise ValueError('Unknown approximate stationarity policy')
    for key in ('force_state_dynamics_source_sha256','force_state_accuracy_source_sha256'):
        h=report.get(key)
        if h is None and not require_sources:
            continue
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('Approximate stationarity policy lacks exact source identities')


def validate_stationarity_accuracy(row,report):
    validate_stationarity_policy(report)
    residual=row.get('dynamic_residual_max_N',float('nan'))
    tolerance=row.get('force_tolerance_N',float('nan'))
    if not np.isfinite(residual) or not np.isfinite(tolerance) or residual<0 or tolerance<=0:
        raise ValueError('Missing finite physical-force residual and positive absolute target')
    evidence=row.get('stationarity_accuracy')
    if evidence is None:
        if residual>tolerance or row.get('approximate_stationarity') is True or row.get('absolute_force_target_passed') is False:
            raise ValueError('Missed absolute force target requires an explicit qualified approximate-stationarity route')
        return {'route':'absolute_force_target','approximate_stationarity':False,'absolute_force_target_passed':True}
    if report.get('stationarity_accuracy_policy')!=POLICY or evidence.get('policy')!=POLICY or evidence.get('qualified') is not True:
        raise ValueError('Missing declared qualified stationarity accuracy policy')
    route=evidence.get('route')
    if route=='absolute_force_target':
        if residual>tolerance or evidence.get('absolute_force_target_N')!=tolerance or \
                evidence.get('absolute_force_target_passed') is not True or row.get('absolute_force_target_passed',True) is not True or \
                row.get('approximate_stationarity') is not False:
            raise ValueError('Absolute route must actually satisfy its unchanged force target')
        return {'route':route,'approximate_stationarity':False,'absolute_force_target_passed':True}
    if route!='full_qp_force_state_fallback' or row.get('approximate_stationarity') is not True or \
            row.get('absolute_force_target_passed') is not False or not residual>tolerance or tolerance!=1e-7:
        raise ValueError('Approximate route must explicitly preserve and disclose the missed 1e-7 N target')
    validate_stationarity_policy(report,require_sources=True)
    for key in ('full_qp_actual_reaction_qualified','qp_solved','actual_comaximality_satisfied',
                'branch_weights_nonnegative','all_physical_contact_and_barrier_reactions_included'):
        if evidence.get(key) is not True:
            raise ValueError(f'Missing full unregularized physical-reaction evidence: {key}')
    if evidence.get('search_only') is not False or evidence.get('numerical_relative_diagonal_regularization')!=0 or \
            evidence.get('residual_projection_scope')!='source_origin_support_reactions_only':
        raise ValueError('A partial/regularized/seam-projected search cannot qualify approximate stationarity')
    quantities={key:evidence.get(key,float('nan')) for key in ('group_weight_sum_error','relative_force_balance',
        'maximum_occupied_site_correction_mm','maximum_occupied_site_velocity_correction_mm_s',
        'source_constraint_direction_error_mm')}
    if any(not np.isfinite(value) or value<0 for value in quantities.values()):
        raise ValueError('Missing finite nonnegative local force/correction evidence')
    if quantities['group_weight_sum_error']>1e-10 or quantities['source_constraint_direction_error_mm']>1e-8 or \
            evidence.get('relative_force_tolerance')!=1e-5 or evidence.get('maximum_occupied_site_correction_tolerance_mm')!=.0005 or \
            quantities['relative_force_balance']>1e-5 or quantities['maximum_occupied_site_correction_mm']>.0005:
        raise ValueError('Approximate stationarity fails its fixed full-force/local-correction contract')
    scale=row.get('force_component_magnitude_sum_max_N',float('nan'))
    dt=row.get('dt_s',float('nan'))
    if not np.isfinite(scale) or scale<=0 or not np.isfinite(dt) or dt<=0 or \
            quantities['relative_force_balance']!=row.get('relative_dynamic_balance_max') or \
            not np.isclose(quantities['relative_force_balance'],residual/scale,rtol=1e-12,atol=1e-15) or \
            not np.isclose(quantities['maximum_occupied_site_velocity_correction_mm_s'],
                quantities['maximum_occupied_site_correction_mm']/dt,rtol=1e-12,atol=1e-15):
        raise ValueError('Local force/velocity indicators do not match the recorded physical residual/time')
    return {'route':route,'approximate_stationarity':True,'absolute_force_target_passed':False,
        'local_indicator_is_not_a_total_state_or_trajectory_error_bound':True}
