"""Qualification within ONE unchanged physical step; only one state is committed.

Default continuation remains available. Opt-in fixed-mu frames claim no fresh
sensitivity unless the immutable schedule requests an actual comparison.
"""
import copy
import time
import numpy as np
from fixed_barrier_policy import (POLICY, CONTINUATION, SCOPE, array_sha,
    accounting_valid, valid_scale, stage_scale)


def barrier_step(dynamics, q_previous, velocity_previous, *, dt, activation,
                 barrier_scales=(1., .2, .04, .008), site_tolerance_mm=.005,
                 stage_callback=None, progress_callback=None, checkpoint_callback=None,
                 converged_stage_q_callback=None, first_initial_iterate=None,
                 max_operation_chunks=1, final_physical_path=None,
                 barrier_policy=CONTINUATION, fixed_barrier_scale=.04,
                 sensitivity_request=None, sensitivity_stage_limit=4,
                 sensitivity_candidate_callback=None, resumed_candidates=(), resumed_partial_stage=None, **step_options):
    fixed = barrier_policy == POLICY
    if barrier_policy not in (CONTINUATION, POLICY):
        raise ValueError('Unknown barrier qualification policy')
    scales = np.asarray(barrier_scales, float)
    if not 0 < site_tolerance_mm <= .005:
        raise ValueError('Occupied-site tolerance <=0.005mm required')
    if fixed:
        if not valid_scale(fixed_barrier_scale) or not isinstance(sensitivity_stage_limit, int) or sensitivity_stage_limit < 2:
            raise ValueError('Declared fixed level and comparison stage budget required')
        if not isinstance(sensitivity_request, dict) or sensitivity_request.get('schema') != 'fixed_barrier_check_request_v1':
            raise ValueError('Fixed mode requires a provenance-bound comparison request')
        check = sensitivity_request.get('comparison_required')
        if not isinstance(check, bool):
            raise ValueError('Explicit comparison selection required')
        scales = np.array([stage_scale(fixed_barrier_scale, i) for i in range(sensitivity_stage_limit if check else 1)])
    else:
        check = True
        if (len(scales) < 2 or not np.isfinite(scales).all() or np.any(scales <= 0) or np.any(np.diff(scales) >= 0)):
            raise ValueError('Strictly decreasing positive barrier scales required')
        if resumed_candidates or resumed_partial_stage is not None:
            raise ValueError('Retained fixed candidates cannot enter legacy continuation')
    if not isinstance(max_operation_chunks, int) or max_operation_chunks < 1:
        raise ValueError('Positive integer operation chunk budget required')
    previous = np.array(q_previous, float, copy=True)
    velocity = np.array(velocity_previous, float, copy=True)
    previous.setflags(write=False); velocity.setflags(write=False)
    started = time.perf_counter()
    warm, stages, comparisons, candidates, inners = first_initial_iterate, [], [], [], []
    selected = None
    candidate = previous.copy()
    if resumed_partial_stage is not None:
        partial = resumed_partial_stage
        i = partial.get('barrier_stage')
        if (i != len(resumed_candidates) or i >= len(scales)
                or partial.get('barrier_policy') != POLICY
                or partial.get('barrier_scale') != scales[i]
                or partial.get('q_previous_sha256') != array_sha(previous)
                or partial.get('velocity_previous_sha256') != array_sha(velocity)
                or partial.get('dt_s') != dt or partial.get('activation') != activation
                or partial.get('check_request') != sensitivity_request
                or np.asarray(partial.get('q')).shape != previous.shape
                or not np.isfinite(partial['q']).all()):
            raise ValueError('Partial fixed stage does not match retained prefix and original history')
    if len(resumed_candidates) > len(scales):
        raise ValueError('Retained stage prefix exceeds this declared operation budget')
    for i, scale in enumerate(scales):
        dynamics.guard.set_barrier_scale(float(scale))
        chunks = []
        if i < len(resumed_candidates):
            saved = resumed_candidates[i]
            if (saved['barrier_scale'] != scale or saved['q_previous_sha256'] != array_sha(previous)
                    or saved['velocity_previous_sha256'] != array_sha(velocity)
                    or saved['dt_s'] != dt or saved['activation'] != activation
                    or saved['check_request'] != sensitivity_request
                    or saved['inner'].get('accepted') is not True):
                raise ValueError('Retained candidate differs from exact original stage/history/check')
            candidate, inner = np.array(saved['q'], copy=True), copy.deepcopy(saved['inner'])
            if (candidate.shape != previous.shape or not np.isfinite(candidate).all()
                    or saved.get('candidate_q_sha256') != array_sha(candidate)
                    or inner.get('contact_accounting', {}).get('barrier_scale') != scale):
                raise ValueError('Retained candidate state/accounting changed')
        else:
            if resumed_partial_stage is not None and i == resumed_partial_stage['barrier_stage']:
                warm = np.array(resumed_partial_stage['q'], copy=True)
                history = resumed_partial_stage.get('failed_inner_report', {}).get('history', [])
                dynamics._epigraph_qp_checkpoint = history[-1].get('epigraph_qp_checkpoint') if history else None
            for chunk in range(max_operation_chunks):
                diagnostic = {}
                candidate, _, inner = dynamics.step(previous, velocity, dt=dt,
                    activation=activation, initial_iterate=warm,
                    defer_physical_frame_path=True,
                    diagnostic_iterate_callback=lambda q, row: diagnostic.update(iterate=q, report=row),
                    progress_callback=(None if progress_callback is None else
                        lambda row: progress_callback(dict(barrier_stage=i, barrier_scale=float(scale),
                            operation_chunk=chunk, **row))), **step_options)
                if inner['accepted']:
                    break
                if checkpoint_callback is not None and diagnostic:
                    checkpoint_callback(diagnostic['iterate'], previous, velocity,
                        dict(barrier_scale=float(scale), barrier_stage=i, dt_s=float(dt), activation=activation,
                            barrier_policy=barrier_policy, check_request=sensitivity_request,
                            q_previous_sha256=array_sha(previous), velocity_previous_sha256=array_sha(velocity),
                            completed_inner_stages=stages, failed_inner_report=inner,
                            prior_operation_chunks=chunks.copy(), operation_chunk=chunk,
                            nonphysical_internal_iterate=True, usable_as_accepted_frame=False))
                chunks.append(dict(inner, not_committed_physical_frame=True))
                history = inner.get('history', [])
                progressing = (len(history) > 1 and history[-1]['implicit_objective_N_mm']
                    < history[0]['implicit_objective_N_mm'])
                resumable = inner['status'] in ('iteration_limit', 'wall_time_budget', 'epigraph_working_set_budget')
                if inner['status'] == 'epigraph_working_set_budget' and history:
                    qp = history[-1].get('epigraph_qp_checkpoint', {})
                    progressing = qp.get('current_model_N_mm', float('inf')) < qp.get('initial_model_N_mm', -float('inf'))
                if not (resumable and progressing and diagnostic and chunk+1 < max_operation_chunks):
                    break
                warm = diagnostic['iterate'].copy()
        stage = copy.deepcopy(inner)
        if chunks:
            stage['operation_chunks'] = chunks
            stage['operation_chunk_count'] = len(chunks)+int(inner['accepted'])
            stage['wall_seconds'] = sum(c.get('wall_seconds', 0.) for c in chunks) + (inner.get('wall_seconds', 0.) if inner['accepted'] else 0.)
        stage['inner_stationarity_converged'] = bool(stage.pop('accepted'))
        stage['not_committed_physical_frame'] = True
        stage['barrier_scale'] = float(scale)
        stage['final_accounting_qualified'] = bool(inner['accepted'] and accounting_valid(inner.get('contact_accounting', {})))
        stages.append(stage)
        if not stage['inner_stationarity_converged'] or not stage['final_accounting_qualified']:
            if stage_callback is not None:
                stage_callback(stage, None)
            break
        candidate = np.array(candidate, copy=True)
        candidates.append(candidate); inners.append(copy.deepcopy(inner))
        if fixed and check and sensitivity_candidate_callback is not None and i >= len(resumed_candidates):
            sensitivity_candidate_callback(candidate.copy(), previous, velocity, dict(
                barrier_policy=POLICY, barrier_stage=i, barrier_scale=float(scale), dt_s=float(dt),
                activation=activation, check_request=sensitivity_request, inner=copy.deepcopy(inner),
                q_previous_sha256=array_sha(previous), velocity_previous_sha256=array_sha(velocity),
                candidate_q_sha256=array_sha(candidate), nonphysical_internal_iterate=True,
                usable_as_accepted_frame=False))
        if converged_stage_q_callback is not None:
            converged_stage_q_callback(candidate.copy(), dict(
                barrier_stage=i, barrier_scale=float(scale), dt_s=float(dt),
                inner_stationarity_converged=True, final_accounting_qualified=True,
                nonphysical_internal_iterate=True, usable_as_accepted_frame=False,
                final_physical_path_certified=False))
        comparison = None
        if i:
            difference = float(dynamics.guard.max_physical_site_difference(candidates[-2], candidate))
            # Backward-Euler velocities share the ORIGINAL physical predecessor.
            # Their dt-scaled difference is exactly the q difference in site space.
            velocity_difference = float(dynamics.guard.max_physical_site_difference(
                (candidates[-2]-previous), (candidate-previous)))
            comparison = dict(previous_scale=float(scales[i-1]), upper_scale=float(scales[i-1]), lower_scale=float(scale),
                maximum_occupied_site_difference_mm=difference,
                dt_scaled_velocity_difference_mm=velocity_difference,
                tolerance_mm=float(site_tolerance_mm), passed=bool(max(difference, velocity_difference) < site_tolerance_mm))
            comparisons.append(comparison)
            if comparison['passed']:
                selected = i-1 if fixed else i
        elif fixed and not check:
            selected = 0
        if stage_callback is not None:
            stage_callback(stage, comparison)
        if selected is not None:
            break
        warm = candidate.copy()
    qualified = selected is not None
    if qualified:
        candidate = candidates[selected]
        inner = copy.deepcopy(inners[selected])
        dynamics.guard.set_barrier_scale(float(scales[selected]))
        if fixed:
            # Restore the selected upper mu before accounting/path. Retain its
            # original full-QP mixture and reactions, never those from lower mu.
            fresh = dict(dynamics.guard.contact_accounting(candidate, include_vectors=False))
            for key in ('quadratic_contact_force_net_N', 'quadratic_contact_force_representation', 'finite_exterior_epigraph'):
                if key in inner['contact_accounting']:
                    fresh[key] = inner['contact_accounting'][key]
            inner['contact_accounting'] = fresh
            qualified = accounting_valid(fresh) and fresh.get('barrier_scale') == float(scales[selected])
    frame_path = (final_physical_path or dynamics.guard.path)(previous, candidate) if qualified else inner.get('frame_path', {})
    path_qualified = bool(qualified and dynamics.path_is_valid(frame_path))
    accepted = bool(qualified and path_qualified)
    report = copy.deepcopy(inner)
    sensitivity_passed = bool(comparisons and comparisons[-1]['passed'] and qualified)
    failure = ('fixed_barrier_sensitivity_inconclusive' if fixed and check else 'barrier_stage_failed')
    if inner['accepted'] and not qualified:
        failure = 'fixed_barrier_sensitivity_budget_exhausted' if fixed else 'barrier_sensitivity_unresolved'
    report.update(accepted=accepted, contact_enforcement='compact_cap_barrier_v1',
        frame_path=frame_path, physical_frame_path_deferred=not qualified,
        final_physical_path_qualified=path_qualified,
        acceptance_scope='physical_frame_with_final_parameter_source_to_endpoint_path',
        status=('converged' if accepted else 'final_physical_path_failed') if qualified else failure,
        wall_seconds=time.perf_counter()-started,
        barrier_continuation_qualified=bool(qualified and not fixed),
        lower_mu_site_difference_mm=comparisons[-1]['maximum_occupied_site_difference_mm'] if comparisons else None,
        lower_mu_site_tolerance_mm=float(site_tolerance_mm), lower_mu_comparison_passed=sensitivity_passed,
        strict_positive_slack=bool(inner['accepted'] and accounting_valid(inner.get('contact_accounting', {}))),
        complementarity_bound_satisfied=bool(inner['accepted'] and accounting_valid(inner.get('contact_accounting', {}))),
        final_inner_stationarity_converged=bool(inner['accepted']))
    common = dict(qualified=bool(qualified), stages=stages, comparisons=comparisons,
        same_physical_time_step=True, dt_s=float(dt), physical_history_unchanged_during_stages=True,
        committed_physical_time_steps=int(accepted),
        accepted_barrier_scale=float(scales[selected]) if qualified else None,
        occupied_site_tolerance_mm=float(site_tolerance_mm))
    if fixed:
        report.update(barrier_policy=POLICY, fixed_barrier_qualified=bool(qualified),
            barrier_qualification=dict(common, policy=POLICY, scope=SCOPE,
                starting_fixed_scale=float(fixed_barrier_scale), selected_stage_index=selected,
                selected_barrier_scale=float(scales[selected]) if qualified else None,
                comparison_performed=bool(comparisons), comparison_required=check,
                comparison_time_s=sensitivity_request['physical_time_s'] if check else None,
                check_request=copy.deepcopy(sensitivity_request),
                q_previous_sha256=array_sha(previous), velocity_previous_sha256=array_sha(velocity),
                inertial_prediction_sha256=array_sha(previous+dt*velocity), activation=copy.deepcopy(activation),
                selected_q_sha256=array_sha(candidate) if qualified else None,
                comparison_stage_limit=sensitivity_stage_limit,
                prospective_scale_reduction=bool(qualified and scales[selected] < fixed_barrier_scale),
                recovery='Preserve original history and retained candidates; no automatic dt or mu reduction on an inconclusive solve'))
    else:
        report['barrier_continuation'] = dict(common,
            sensitivity_scope='Successive lower-mu occupied-site comparison, not a rigorous total discretization-error bound')
    if qualified and not path_qualified and checkpoint_callback is not None:
        checkpoint_callback(candidate, previous, velocity,
            dict(barrier_scale=float(scales[selected]), barrier_stage=selected,
                dt_s=float(dt), activation=activation, barrier_policy=barrier_policy,
                completed_inner_stages=stages, failed_inner_report=report,
                nonphysical_internal_iterate=True, usable_as_accepted_frame=False,
                final_parameter_stationary_but_physical_path_failed=True))
    if not accepted:
        return previous.copy(), velocity.copy(), report
    return candidate, (candidate-previous)/dt, report
