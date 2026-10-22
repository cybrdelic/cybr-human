"""Matched whole-face coarse-site dynamics with honest cap-reaction continuation."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from barrier_continuation import barrier_step
from clipped_origin_support import load_support_bundle
from first_step import save_accepted_frames
from implicit_dynamics import ImplicitDynamics, source_activation
from source_transfer import load_geometry_with_source_fibres, sha
from accepted_history import next_physical_time, validate_accepted_history, validate_resume_interval_audits, matched_passive_numerical_guess, inertia_scaled_numerical_guess


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--guard-module', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dt', type=float, default=.01)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--wall-seconds-per-stage', type=float, default=240.)
    p.add_argument('--operation-chunks', type=int, default=1)
    p.add_argument('--force-tolerance-N', type=float, default=1e-7)
    p.add_argument('--ramp-s', type=float, default=.1)
    p.add_argument('--barrier-scales', nargs='+', type=float, default=[1., .2, .04, .008])
    p.add_argument('--cases', nargs='+', choices=['passive', 'zygomaticus_frontalis'],
        default=['passive', 'zygomaticus_frontalis'])
    p.add_argument('--resume-directory', type=Path)
    p.add_argument('--prior-interval-audits', nargs='+', type=Path, default=[],
        help='Hash-bound passing external continuous site/J audits of the accepted history')
    p.add_argument('--previous-guard-sha256', help='Explicit reviewed law-preserving rolling/finite-branch implementation transition from this exact saved base hash')
    p.add_argument('--stop-after-failed-case', action='store_true',
        help='Pause at a saved internal checkpoint rather than cold-starting the other case')
    p.add_argument('--nonphysical-warm-start', type=Path,
        help='Explicit same-step internal checkpoint; never an accepted frame or new physical history')
    p.add_argument('--inertia-scaled-diagnostic-guess', type=Path,
        help='Q-only acceleration-scaled guess from a failed stationary endpoint, with identical prior q/v and a NEW shorter physical dt; never a frame')
    a = p.parse_args()
    if not 0 < a.dt <= .02 or not 1 <= a.steps <= 100 or not 1 <= a.iterations <= 40 or not 0 < a.wall_seconds_per_stage <= 600:
        p.error('Bounded positive dt<=0.02s, steps<=100, iterations<=40, wall per stage<=600s required')
    if a.output.exists():
        p.error('New output directory required; preserve earlier results')
    if a.inertia_scaled_diagnostic_guess is not None and a.nonphysical_warm_start is not None:
        p.error('A different-dt numerical guess cannot resume the old same-step QP')
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    domain = here.parent/'geometry/whole-face-continuum-final.npz'
    phases = here.parent/'geometry/whole-face-muscle-phases-final.npz'
    solid = load_geometry_with_source_fibres(domain, phases)
    supports, support_report = load_support_bundle(solid, here/'coarse-physical-site-v1/supports')
    prior = None if a.resume_directory is None else json.loads((a.resume_directory/'report.json').read_text())
    report = dict(status='preparing_coarse_barrier_contact', contact_mode='coarse_physical_sites',
        guard_scope='coarse_physical_sites', contact_enforcement='compact_cap_barrier_v1',
        global_surface_collision_certified=False, site_resolution_mm=1.,
        source_domain_sha256=sha(domain), source_phase_sha256=sha(phases),
        physical_dt_s=a.dt, requested_steps_per_case=a.steps, activation_ramp_s=a.ramp_s,
        physical_dt_s_scope='Newly requested intervals only; prior actual dt is retained per frame',
        time_policy='Cumulative accepted physical time; each case retains its own q and velocity',
        physical_mass_kg=float(solid.mass.sum()*1000), damping_per_second=0., gravity_mm_s2=[0., -9810., 0.],
        initial_state='Registered source, non-equilibrated transient settling', initial_velocity='zero',
        length_scale='1L0=1mm is an explicit engineering assumption, not patient calibration',
        model_resolution='6-L0 P1 carrier and positive1-L0 occupied physical sites; no exact render-boundary collision certification',
        source_origin_support=support_report, accepted_facial_frames=0, exact_boundary_certified_frames=0,
        source_support_sha256=sha(here/'coarse-physical-site-v1/supports/clipped-origin-support.json'),
        source_support_bundle_sha256={f.name: sha(f) for f in sorted((here/'coarse-physical-site-v1/supports').glob('*.npz'))},
        lower_mu_site_tolerance_mm=.005, barrier_scales=a.barrier_scales, runs={})
    def persist():
        report['wall_seconds'] = time.perf_counter()-started
        temporary = a.output/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(a.output/'report.json')
    persist()
    if prior is not None:
        for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256', 'activation_ramp_s', 'contact_mode', 'contact_enforcement'):
            if prior.get(key) != report[key]:
                raise ValueError('Continuation model differs: '+key)
        if prior.get('source_support_bundle_sha256', report['source_support_bundle_sha256']) != report['source_support_bundle_sha256']:
            raise ValueError('Continuation source-support material bundle differs')
        report.update(resumed_report_path=str((a.resume_directory/'report.json').resolve()),
            resumed_report_sha256=sha(a.resume_directory/'report.json'))
        report['prior_continuous_interval_audits'] = validate_resume_interval_audits(prior, a.prior_interval_audits, a.cases)
        if any(a.steps > prior['runs'].get(case, {}).get('accepted_frames', 0)+1 for case in a.cases):
            raise ValueError('Advance at most one new interval per case before its external continuous audit')
    try:
        spec = importlib.util.spec_from_file_location('coarse_barrier_guard', a.guard_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        guard = module.create_guard(solid, supports)
        if guard.provenance.get('contact_enforcement') != 'compact_cap_barrier_v1':
            raise ValueError('The explicitly qualified barrier adapter is required')
        implementation_transition = None
        if prior is not None:
            if prior['coarse_contact_provenance'].get('rigid_query_source_sha256') != guard.provenance.get('rigid_query_source_sha256'):
                raise ValueError('Continuation native contact query source differs')
            if prior['guard_module_sha256'] != sha(a.guard_module):
                if not (a.previous_guard_sha256 == prior['guard_module_sha256']
                        == guard.provenance.get('base_contact_source_sha256')
                        and guard.provenance.get('original_minimum_distance_energy_unchanged') is True
                        and guard.provenance.get('exterior_branch_representation') == 'complete_exterior_finite_energy_branches_v1'
                        and guard.provenance.get('exterior_branch_provider_sha256') == sha(a.guard_module)):
                    raise ValueError('Unqualified contact implementation transition')
                implementation_transition = dict(previous_guard_sha256=a.previous_guard_sha256,
                    current_guard_sha256=sha(a.guard_module),
                    scope='Reviewed exact rolling query acceleration and finite max-energy branch representation; base physical law/guards unchanged',
                    source_history_dt_mu_support_mass_unchanged=True)
        report.update(guard_module=str(a.guard_module.resolve()), guard_module_sha256=sha(a.guard_module),
            coarse_contact_provenance=guard.provenance, guard_capabilities=guard.certification)
        if implementation_transition is not None:
            report['reviewed_implementation_transition'] = implementation_transition
        if hasattr(guard, 'contact_branches'):
            report['exterior_contact_stationarity'] = 'finite_exterior_energy_epigraph_v1'
            report['epigraph_mechanics_source_sha256'] = sha(here/'exterior_epigraph.py')
            report['epigraph_linear_adapter_source_sha256'] = sha(here/'factored_seam_kkt.py')
            report['epigraph_compact_rank_source_sha256'] = sha(here/'compact_seam_rank.py')
            report['epigraph_linear_method'] = 'Exact source-KKT factor reuse and changing-seam Schur complement; physical equations unchanged'
        if prior is not None:
            for key in ('clearance_mm', 'penalty_stiffness_N_per_mm4', 'maximum_allowed_site_penetration_mm', 'guard_scope'):
                if prior['coarse_contact_provenance'].get(key) != guard.provenance.get(key):
                    raise ValueError('Continuation physical contact parameter differs: '+key)
        checkpoint = None
        scaled_guess = None
        if a.inertia_scaled_diagnostic_guess is not None:
            old = json.loads(a.inertia_scaled_diagnostic_guess.with_suffix('.json').read_text())
            if sha(a.inertia_scaled_diagnostic_guess) != old['npz_sha256'] or old.get('nonphysical_internal_iterate') is not True:
                raise ValueError('Different-dt initial guess is not a verified NONPHYSICAL checkpoint')
            for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256', 'guard_module_sha256'):
                if old[key] != report[key]:
                    raise ValueError('Different-dt guess source/physics mismatch: '+key)
            if not 0 < a.dt < old['dt_s'] or old.get('final_parameter_stationary_but_physical_path_failed') is not True:
                raise ValueError('Different-dt guess requires a genuinely shorter step and a stationary failed physical endpoint')
            own_previous,own_velocity,history_time=solid.X,np.zeros_like(solid.X),0.
            history_sha=None
            if prior is not None:
                history=prior['runs'][old['case']]
                if history['accepted_frames']:
                    history_sha=history['accepted_npz_sha256']
                    if sha(history['accepted_npz_path'])!=history_sha:
                        raise ValueError('Different-dt guess prior accepted file changed')
                    with np.load(history['accepted_npz_path'],allow_pickle=False) as z:
                        own_previous=z['positions_mm'][-1].copy();own_velocity=z['velocities_mm_s'][-1].copy()
                        history_time=float(z['physical_times_s'][-1])
            if not np.isclose(old['physical_time_s']-old['dt_s'],history_time,rtol=1e-12,atol=1e-14):
                raise ValueError('Different-dt guess must start at the same accepted physical time')
            with np.load(a.inertia_scaled_diagnostic_guess, allow_pickle=False) as z:
                if not np.array_equal(z['q_previous_mm'],own_previous) or not np.array_equal(z['velocity_previous_mm_s'],own_velocity):
                    raise ValueError('Different-dt guess prior q/velocity differs from its own accepted history')
                factor = (a.dt/old['dt_s'])**2
                candidate = inertia_scaled_numerical_guess(own_previous,own_velocity,z['nonphysical_iterate_mm'],old['dt_s'],a.dt)
            endpoint = guard.endpoint(candidate)
            usable = bool(endpoint['feasible'] and supports.error(candidate) <= 1e-8)
            scaled_guess = (old['case'], candidate, history_time+a.dt) if usable else None
            report['different_dt_numerical_guess'] = dict(source_path=str(a.inertia_scaled_diagnostic_guess.resolve()),
                source_npz_sha256=old['npz_sha256'], old_dt_s=old['dt_s'], new_dt_s=a.dt,
                acceleration_correction_factor=factor, actual_endpoint=endpoint, used=usable,
                prior_history_npz_sha256=history_sha, previous_physical_time_s=history_time,
                formula='q_previous + dt_new*v_previous + (dt_new/dt_old)^2*(q_failed-q_previous-dt_old*v_previous)',
                physical_history='Own accepted q/velocity; fixed NEW physical dt and inertial prediction',
                numerical_guess_only=True, interpolated_motion_frame=False)
        if a.nonphysical_warm_start is not None:
            checkpoint = json.loads(a.nonphysical_warm_start.with_suffix('.json').read_text())
            if sha(a.nonphysical_warm_start) != checkpoint['npz_sha256'] or checkpoint.get('nonphysical_internal_iterate') is not True:
                raise ValueError('Invalid internal checkpoint identity')
            for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256', 'guard_module_sha256'):
                expected = a.previous_guard_sha256 if key == 'guard_module_sha256' and implementation_transition is not None else report[key]
                if checkpoint[key] != expected:
                    raise ValueError('Internal checkpoint model mismatch: '+key)
            if checkpoint['dt_s'] != a.dt or checkpoint['barrier_scale'] != a.barrier_scales[0]:
                raise ValueError('Internal checkpoint must resume the SAME physical timestep and mu')
            report['nonphysical_warm_start_source'] = str(a.nonphysical_warm_start.resolve())
        dynamics = ImplicitDynamics(solid, guard, constraints=supports,
            guard_scope='coarse_physical_sites', damping_per_second=0.)
        dynamics.require_guards()
        report['status'] = 'running_coarse_barrier_transient'
        persist()
        passive_initial_guesses = {}
        for case in a.cases:
            q, v = solid.X.copy(), np.zeros_like(solid.X)
            frames, velocities, times, rows = [], [], [], []
            if prior is not None:
                old = prior['runs'].get(case, {'accepted_frames': 0})
                if old['accepted_frames']:
                    saved = Path(old['accepted_npz_path'])
                    if sha(saved) != old['accepted_npz_sha256']:
                        raise ValueError('Saved accepted state hash mismatch')
                    with np.load(saved, allow_pickle=False) as z:
                        if not np.array_equal(z['source_initial_positions_mm'], solid.X):
                            raise ValueError('Source reference mismatch')
                        frames = list(z['positions_mm'].copy())
                        velocities = list(z['velocities_mm_s'].copy())
                        times = z['physical_times_s'].tolist()
                    rows = [r for r in old['reports'] if r['accepted']]
                    if len(rows) != len(frames) or times != old['actual_times_s']:
                        raise ValueError('Accepted history mismatch')
                    validate_accepted_history(solid.X, frames, velocities, times, rows)
                    q, v = frames[-1].copy(), velocities[-1].copy()
                    report['accepted_facial_frames'] += len(frames)
                    if case == 'passive':
                        passive_initial_guesses.update(zip(times, frames))
            run = dict(accepted_frames=len(frames), actual_times_s=times, reports=rows)
            report['runs'][case] = run
            if frames:
                run.update(**save_accepted_frames(a.output/f'{case}-accepted.npz', positions=frames,
                    velocities=velocities, times=times, solid=solid))
            persist()
            for step in range(len(frames)+1, a.steps+1):
                physical_time = next_physical_time(times, a.dt)
                dynamics._epigraph_qp_checkpoint = None
                print(json.dumps(dict(case=case, step=step, stage='step_start', physical_time_s=physical_time)), flush=True)
                run['pending_barrier_stages'] = []
                warm = matched_passive_numerical_guess(case, physical_time, passive_initial_guesses, q, guard, supports)
                warm_source = 'matched-time passive q only; active physical history unchanged' if warm is not None else 'own previous physical state'
                if scaled_guess is not None and scaled_guess[0] == case and np.isclose(physical_time,scaled_guess[2],rtol=1e-12,atol=1e-14):
                    warm = scaled_guess[1].copy()
                    warm_source = 'inertia-scaled NONPHYSICAL larger-dt diagnostic used only as Newton guess; own accepted physical history/prediction unchanged'
                if checkpoint is not None and checkpoint['case'] == case and checkpoint['physical_time_s'] == physical_time:
                    with np.load(a.nonphysical_warm_start, allow_pickle=False) as z:
                        if not np.array_equal(z['q_previous_mm'], q) or not np.array_equal(z['velocity_previous_mm_s'], v):
                            raise ValueError('Internal checkpoint physical history differs')
                        warm = z['nonphysical_iterate_mm'].copy()
                    warm_source = 'same-step nonphysical internal checkpoint; physical history unchanged'
                    inner_history = checkpoint.get('failed_inner_report', {}).get('history', [])
                    if inner_history:
                        dynamics._epigraph_qp_checkpoint = inner_history[-1].get('epigraph_qp_checkpoint')
                    if checkpoint['activation'] != source_activation(physical_time, case=case, ramp_s=a.ramp_s):
                        raise ValueError('Internal checkpoint activation differs')
                def save_checkpoint(iterate, previous, velocity, metadata):
                    path = a.output/f'{case}-step{step:04d}-NONPHYSICAL-checkpoint.npz'
                    np.savez_compressed(path, nonphysical_iterate_mm=iterate,
                        q_previous_mm=previous, velocity_previous_mm_s=velocity)
                    metadata.update(case=case, physical_time_s=physical_time, npz_sha256=sha(path),
                        **{key: report[key] for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256', 'guard_module_sha256')})
                    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
                    run['nonphysical_checkpoint_path'] = str(path.resolve())
                def stage_finished(stage, comparison):
                    run['pending_barrier_stages'].append(dict(scale=stage['barrier_scale'],
                        status=stage['status'], dynamic_residual_max_N=stage['dynamic_residual_max_N'],
                        wall_seconds=stage['wall_seconds'], comparison=comparison))
                    persist()
                    print(json.dumps(dict(case=case, step=step, stage='barrier_stage_finished',
                        **run['pending_barrier_stages'][-1])), flush=True)
                q, v, row = barrier_step(dynamics, q, v, dt=a.dt,
                    activation=source_activation(physical_time, case=case, ramp_s=a.ramp_s),
                    barrier_scales=a.barrier_scales, site_tolerance_mm=.005,
                    max_iterations=a.iterations, max_wall_seconds=a.wall_seconds_per_stage,
                    force_tolerance_N=a.force_tolerance_N, stage_callback=stage_finished,
                    first_initial_iterate=warm, checkpoint_callback=save_checkpoint,
                    max_operation_chunks=a.operation_chunks,
                    progress_callback=lambda state: print(json.dumps(dict(case=case, step=step, stage='Newton', **state)), flush=True))
                row.update(physical_time_s=physical_time,
                    newton_initial_guess_source=warm_source,
                    carrier_displacement_max_mm=float(np.linalg.norm(q-solid.X, axis=1).max()))
                rows.append(row)
                run.pop('pending_barrier_stages', None)
                if row['accepted']:
                    frames.append(q.copy()); velocities.append(v.copy()); times.append(physical_time)
                    report['accepted_facial_frames'] += 1
                    if case == 'passive':
                        passive_initial_guesses[physical_time] = q.copy()
                    run.update(accepted_frames=len(frames), actual_times_s=times,
                        **save_accepted_frames(a.output/f'{case}-accepted.npz', positions=frames,
                            velocities=velocities, times=times, solid=solid))
                persist()
                print(json.dumps(dict(case=case, step=step, status=row['status'], accepted=row['accepted'],
                    wall_seconds=row['wall_seconds'], dynamic_residual_N=row['dynamic_residual_max_N'],
                    lower_mu_site_difference_mm=row['lower_mu_site_difference_mm'], minimum_J=row['minimum_J'])), flush=True)
                if not row['accepted']:
                    break
            if rows and not rows[-1]['accepted'] and a.stop_after_failed_case:
                break
        report['status'] = 'bounded_barrier_attempt_finished'
    except Exception as exc:
        report.update(status='barrier_attempt_error', error_type=type(exc).__name__, blocker=str(exc))
        persist()
        raise
    persist()
    print(json.dumps(dict(status=report['status'], accepted_facial_frames=report['accepted_facial_frames'],
        wall_seconds=report['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
