"""Versioned fixed barrier approximation and immutable empirical-check schedule.

No force, geometry, nonlinear stationarity or integration equations live here.
"""
import hashlib
import json
import math
import numpy as np

POLICY = 'fixed_mu_0p04_empirical_promotion_v1'
CONTINUATION = 'successive_lower_mu_v1'
FIXED_SCALE = .04
MODEL = dict(barrier='mu*(-log(s/s0)+s/s0-1) for 0<s<s0, zero for s>=s0',
    slack='s=g+p', penetration_limit_mm=.05, compact_band_s0_mm=.01,
    base_mu='k*V_i*(h+p)*s0', penalty_stiffness_k=.01, clearance_h_mm=.1,
    initial_scale=.04, initial_one_mm3_owner_complementarity_bound_N_mm=6e-7)
COMPARISON_SCALE = .008
TOLERANCE_MM = .005
SCOPE = ('Initially fixed .04 barrier approximation, prospectively reduced only after a failed scheduled comparison. Same-history checks '
         'are empirical sensitivity indicators only, not total-error bounds or '
         'certification of untested frames.')


def stage_scale(start, index):
    return float(start / (5 ** index))


def valid_scale(scale):
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= FIXED_SCALE:
        return False
    level = round(math.log(FIXED_SCALE / scale, 5))
    return math.isclose(scale, stage_scale(FIXED_SCALE, level), rel_tol=1e-12, abs_tol=0.)


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def plan_sha(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def retained_model_sha(report, material_provenance, implementation_sha256):
    # Reuse the already reviewed normalization: exclude only named setup timings.
    from curvature_transition import physical_provenance
    model = dict(contact=physical_provenance(report['coarse_contact_provenance']),
        material_source=material_provenance, implementations=implementation_sha256,
        parameters={key: report[key] for key in (
            'source_domain_sha256', 'source_phase_sha256', 'source_support_sha256',
            'source_support_bundle_sha256', 'physical_mass_kg', 'damping_per_second',
            'gravity_mm_s2', 'activation_ramp_s', 'stationarity_accuracy_policy')})
    return plan_sha(model)


def validate_plan(plan):
    if (plan.get('schema') != 'fixed_barrier_check_plan_v1' or plan.get('policy') != POLICY
            or plan.get('initial_fixed_scale') != FIXED_SCALE or plan.get('initial_comparison_scale') != COMPARISON_SCALE
            or plan.get('comparison_divisor') != 5
            or plan.get('empirical_indicator_mm') != TOLERANCE_MM
            or plan.get('trigger') != 'first_actual_interval_crossing_target'):
        raise ValueError('Unknown or changed fixed barrier check plan')
    checks = plan.get('checks', [])
    times = [r.get('target_time_s') for r in checks]
    if (not times or any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in times)
            or times != sorted(set(times)) or not all(r.get('reason') for r in checks)
            or not set((.025, .05, .1)).issubset(times)):
        raise ValueError('Finite ordered checks including 25/50/100 ms required')
    return plan


def make_plan(prior=None, additional_times=(), reason=None, previous_time=0.):
    """Retain old selected checks on resume; add only explicit future checks."""
    old = None if prior is None else prior.get('fixed_barrier_check_plan')
    if old is not None:
        validate_plan(old)
        if prior.get('fixed_barrier_check_plan_sha256') != plan_sha(old):
            raise ValueError('Previous fixed barrier schedule hash changed')
        plan = json.loads(json.dumps(old))
    else:
        plan = dict(schema='fixed_barrier_check_plan_v1', policy=POLICY,
            initial_fixed_scale=FIXED_SCALE, initial_comparison_scale=COMPARISON_SCALE, comparison_divisor=5,
            empirical_indicator_mm=TOLERANCE_MM, trigger='first_actual_interval_crossing_target',
            checks=[dict(target_time_s=t, reason='declared_milestone') for t in (.025, .05, .1)],
            predecessor_plan_sha256=None)
    if additional_times:
        if not reason or any(not math.isfinite(t) or t <= previous_time for t in additional_times):
            raise ValueError('Explicit reason and future times required for added regime checks')
        existing = {r['target_time_s'] for r in plan['checks']}
        additions = [dict(target_time_s=float(t), reason=reason) for t in sorted(set(additional_times)-existing)]
        if additions:
            plan['predecessor_plan_sha256'] = plan_sha(plan)
            plan['checks'] = sorted(plan['checks']+additions, key=lambda r: r['target_time_s'])
    return validate_plan(plan)


def check_request(plan, previous_time, physical_time):
    validate_plan(plan)
    if not (math.isfinite(previous_time) and math.isfinite(physical_time) and 0 <= previous_time < physical_time):
        raise ValueError('Finite increasing physical interval required')
    crossed = [r for r in plan['checks'] if previous_time+1e-14 < r['target_time_s'] <= physical_time+1e-14]
    return dict(schema='fixed_barrier_check_request_v1', plan_sha256=plan_sha(plan),
        comparison_required=bool(crossed), crossed_checks=crossed, plan=json.loads(json.dumps(plan)),
        previous_physical_time_s=float(previous_time), physical_time_s=float(physical_time),
        trigger='first_actual_interval_crossing_target')


def accounting_valid(accounting):
    slack = accounting.get('minimum_slack_mm')
    positive = ((slack is None and accounting.get('active_barrier_owner_terms') == 0)
                or (isinstance(slack, (int, float)) and math.isfinite(slack) and slack > 0))
    return bool(positive and accounting.get('complementarity_bound_satisfied') is True)


def validate_fixed_frame(row, previous=None, velocity=None, candidate=None):
    """Additional policy checks only; all existing geometric/force gates remain."""
    if row.get('barrier_policy') != POLICY:
        raise ValueError('Unknown fixed barrier frame policy')
    q = row.get('barrier_qualification', {})
    request = q.get('check_request', {})
    if (row.get('fixed_barrier_qualified') is not True or q.get('qualified') is not True
            or q.get('policy') != POLICY or not valid_scale(q.get('selected_barrier_scale'))
            or q.get('same_physical_time_step') is not True
            or q.get('physical_history_unchanged_during_stages') is not True
            or q.get('committed_physical_time_steps') != 1
            or not valid_scale(q.get('starting_fixed_scale'))
            or row.get('barrier_continuation_qualified') is not False
            or row.get('final_inner_stationarity_converged') is not True
            or row.get('final_physical_path_qualified') is not True
            or row.get('strict_positive_slack') is not True
            or row.get('complementarity_bound_satisfied') is not True
            or row.get('dynamic_residual_includes_barrier_reaction') is not True
            or row.get('contact_generalized_stationarity_qualified') is not True
            or row.get('physical_frame_path_deferred') is not False
            or row.get('acceptance_scope') != 'physical_frame_with_final_parameter_source_to_endpoint_path'):
        raise ValueError('Missing fixed barrier frame qualification')
    accounting = row.get('contact_accounting', {})
    if accounting.get('barrier_scale') != q['selected_barrier_scale'] or not accounting_valid(accounting):
        raise ValueError('Selected fixed scale/accounting mismatch')
    validate_plan(request.get('plan', {}))
    expected_request = check_request(request['plan'], request['previous_physical_time_s'], request['physical_time_s'])
    if request != expected_request:
        raise ValueError('Comparison request differs from its immutable plan')
    performed = q.get('comparison_performed')
    if (not isinstance(performed, bool) or request.get('comparison_required') != performed
            or request.get('schema') != 'fixed_barrier_check_request_v1'
            or not isinstance(request.get('plan_sha256'), str) or len(request['plan_sha256']) != 64
            or not np.isclose(request.get('physical_time_s', np.nan), row['physical_time_s'], rtol=1e-12, atol=1e-14)
            or not np.isclose(request.get('physical_time_s', np.nan)-request.get('previous_physical_time_s', np.nan), row['dt_s'], rtol=1e-12, atol=1e-14)):
        raise ValueError('Fixed barrier comparison schedule mismatch')
    stages = q.get('stages', [])
    scales = [stage_scale(q['starting_fixed_scale'], i) for i in range(len(stages))]
    selected = len(stages)-2 if performed else 0
    if ((performed and len(stages) < 2) or (not performed and len(stages) != 1)
            or q.get('selected_stage_index') != selected
            or q['selected_barrier_scale'] != scales[selected]):
        raise ValueError('Wrong fixed barrier stage count')
    for stage, scale in zip(stages, scales):
        if (stage.get('barrier_scale') != scale or stage.get('inner_stationarity_converged') is not True
                or stage.get('final_accounting_qualified') is not True
                or stage.get('not_committed_physical_frame') is not True
                or stage.get('physical_frame_path_deferred') is not True
                or stage.get('acceptance_scope') != 'internal_same_timestep_stationarity_only'
                or stage.get('dynamic_residual_includes_barrier_reaction') is not True
                or stage.get('contact_generalized_stationarity_qualified') is not True):
            raise ValueError('Missing same-history fixed barrier stage evidence')
    comparisons = q.get('comparisons', [])
    if performed:
        if len(comparisons) != len(stages)-1 or row.get('lower_mu_comparison_passed') is not True:
            raise ValueError('Scheduled empirical comparison did not qualify')
        for i, comparison in enumerate(comparisons):
            if (comparison.get('upper_scale') != scales[i] or comparison.get('lower_scale') != scales[i+1]
                    or comparison.get('tolerance_mm') != TOLERANCE_MM):
                raise ValueError('Empirical comparison scales differ')
            values = [comparison.get(k) for k in ('maximum_occupied_site_difference_mm', 'dt_scaled_velocity_difference_mm')]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('Invalid empirical sensitivity')
            passed = max(values) < TOLERANCE_MM
            if comparison.get('passed') != passed or passed != (i == len(comparisons)-1):
                raise ValueError('Selected candidate is not the first passing upper stage')
        if row.get('lower_mu_site_difference_mm') != comparisons[-1]['maximum_occupied_site_difference_mm']:
            raise ValueError('Sensitivity summary differs')
    elif (comparisons or row.get('lower_mu_comparison_passed') is not False
            or row.get('lower_mu_site_difference_mm') is not None):
        raise ValueError('Untested frame claims a fresh lower-mu comparison')
    for value, key in ((previous, 'q_previous_sha256'), (velocity, 'velocity_previous_sha256'),
                       (candidate, 'selected_q_sha256')):
        if value is not None and array_sha(value) != q.get(key):
            raise ValueError('Fixed barrier own-history/candidate provenance differs: '+key)
    if previous is not None and velocity is not None:
        if array_sha(np.asarray(previous)+row['dt_s']*np.asarray(velocity)) != q.get('inertial_prediction_sha256'):
            raise ValueError('Fixed barrier own inertial prediction differs')
    return q


def prospective_fixed_scale(report):
    """Lower common level only from admitted fixed-policy states, not check candidates."""
    scales = [.04]
    for run in report.get('runs', {}).values():
        for row in run.get('reports', []):
            if row.get('accepted') is True and row.get('barrier_policy') == POLICY:
                scales.append(validate_fixed_frame(row)['selected_barrier_scale'])
    return min(scales)
