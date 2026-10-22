"""Primal barrier continuation inside ONE unchanged physical time step.

The lower-mu site comparison measures regularization sensitivity, not a rigorous
bound on the total anatomical, spatial, or temporal discretization error.
Only the final qualified state may be committed as a physical frame.
"""
import time
import numpy as np


def barrier_step(dynamics, q_previous, velocity_previous, *, dt, activation,
                 barrier_scales=(1., .2, .04, .008), site_tolerance_mm=.005,
                 stage_callback=None, progress_callback=None, checkpoint_callback=None,
                 first_initial_iterate=None, max_operation_chunks=1, **step_options):
    scales = np.asarray(barrier_scales, float)
    if (len(scales) < 2 or not np.isfinite(scales).all() or np.any(scales <= 0)
            or np.any(np.diff(scales) >= 0) or not 0 < site_tolerance_mm <= .005):
        raise ValueError('Strictly decreasing positive barrier scales and site tolerance <=0.005mm required')
    if not isinstance(max_operation_chunks, int) or max_operation_chunks < 1:
        raise ValueError('Positive integer operation chunk budget required')
    previous = np.array(q_previous, float, copy=True)
    velocity = np.array(velocity_previous, float, copy=True)
    started = time.perf_counter()
    warm, stages, comparisons, qualified = first_initial_iterate, [], [], False
    previous_converged = None
    candidate = previous.copy()
    for i, scale in enumerate(scales):
        dynamics.guard.set_barrier_scale(float(scale))
        chunks = []
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
                progressing = (qp.get('current_model_N_mm', float('inf')) < qp.get('initial_model_N_mm', -float('inf')))

            if not (resumable and progressing and diagnostic and chunk+1 < max_operation_chunks):
                break
            # This is an operation-budget checkpoint, NOT a new timestep or mu.
            # In particular previous_converged remains the prior qualified mu.
            warm = diagnostic['iterate'].copy()
        stage = dict(inner)
        if chunks:
            stage['operation_chunks'] = chunks
            stage['operation_chunk_count'] = len(chunks)+int(inner['accepted'])
            stage['wall_seconds'] = sum(c.get('wall_seconds', 0.) for c in chunks) + (inner.get('wall_seconds', 0.) if inner['accepted'] else 0.)
        stage['inner_stationarity_converged'] = bool(stage.pop('accepted'))
        stage['not_committed_physical_frame'] = True
        stage['barrier_scale'] = float(scale)
        stages.append(stage)
        if not stage['inner_stationarity_converged']:
            if stage_callback is not None:
                stage_callback(stage, None)
            break
        accounting = inner.get('contact_accounting', {})
        slack = accounting.get('minimum_slack_mm')
        positive_slack = (slack is None and accounting.get('active_barrier_owner_terms') == 0) or (slack is not None and slack > 0)
        valid_accounting = positive_slack and accounting.get('complementarity_bound_satisfied') is True
        if not valid_accounting:
            stage['final_accounting_qualified'] = False
            break
        comparison = None
        if previous_converged is not None:
            difference = float(dynamics.guard.max_physical_site_difference(previous_converged, candidate))
            comparison = dict(previous_scale=float(scales[i-1]), lower_scale=float(scale),
                maximum_occupied_site_difference_mm=difference,
                tolerance_mm=float(site_tolerance_mm), passed=bool(difference < site_tolerance_mm))
            comparisons.append(comparison)
            qualified = comparison['passed']
        if stage_callback is not None:
            stage_callback(stage, comparison)
        if qualified:
            break
        warm = candidate.copy()
        previous_converged = candidate.copy()
    frame_path = dynamics.guard.path(previous, candidate) if qualified else inner.get('frame_path', {})
    path_qualified = bool(qualified and dynamics.path_is_valid(frame_path))
    accepted = bool(qualified and path_qualified)
    report = dict(inner)
    report.update(accepted=accepted, contact_enforcement='compact_cap_barrier_v1',
        frame_path=frame_path, physical_frame_path_deferred=not qualified,
        final_physical_path_qualified=path_qualified,
        acceptance_scope='physical_frame_with_final_parameter_source_to_endpoint_path',
        status=('converged' if accepted else 'final_physical_path_failed') if qualified else ('barrier_stage_failed' if not inner['accepted'] else 'barrier_sensitivity_unresolved'),
        wall_seconds=time.perf_counter()-started,
        barrier_continuation=dict(qualified=bool(qualified), stages=stages, comparisons=comparisons,
            same_physical_time_step=True, dt_s=float(dt), physical_history_unchanged_during_stages=True,
            committed_physical_time_steps=int(accepted),
            accepted_barrier_scale=float(scales[len(stages)-1]) if qualified else None,
            occupied_site_tolerance_mm=float(site_tolerance_mm),
            sensitivity_scope='Successive lower-mu occupied-site comparison, not a rigorous total discretization-error bound'),
        barrier_continuation_qualified=bool(qualified),
        lower_mu_site_difference_mm=comparisons[-1]['maximum_occupied_site_difference_mm'] if comparisons else None,
        lower_mu_site_tolerance_mm=float(site_tolerance_mm),
        lower_mu_comparison_passed=bool(qualified),
        strict_positive_slack=bool(positive_slack) if inner['accepted'] else False,
        complementarity_bound_satisfied=bool(valid_accounting) if inner['accepted'] else False,
        final_inner_stationarity_converged=bool(inner['accepted']))
    if qualified and not path_qualified and checkpoint_callback is not None:
        checkpoint_callback(candidate, previous, velocity,
            dict(barrier_scale=float(scales[len(stages)-1]), barrier_stage=len(stages)-1,
                dt_s=float(dt), activation=activation, completed_inner_stages=stages,
                failed_inner_report=report, nonphysical_internal_iterate=True,
                usable_as_accepted_frame=False, final_parameter_stationary_but_physical_path_failed=True))
    if not accepted:
        return previous, velocity, report
    # In particular, do not add velocities from separate barrier stages.
    return candidate, (candidate-previous)/dt, report
