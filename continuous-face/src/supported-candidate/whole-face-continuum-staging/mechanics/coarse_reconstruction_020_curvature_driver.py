"""Matched whole-face coarse-site dynamics with honest cap-reaction continuation."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import os
from candidate_interval_audit import audit_candidate_case
import numpy as np
from barrier_reconstruction_continuation import barrier_step
from fixed_barrier_policy import (POLICY as FIXED_POLICY, CONTINUATION, SCOPE as FIXED_SCOPE, MODEL as FIXED_MODEL,
    validate_plan, plan_sha, check_request, valid_scale, array_sha, validate_fixed_frame, prospective_fixed_scale, stage_scale, retained_model_sha)
from passive_stage_seed import FirstPassiveStageCapture, StageSeedRejected, audited_first_passive_stage_guess
from reconstruction_020_seed import audited_passive_endpoints, RECONSTRUCTION_FIELDS
from reconstruction_020_frame_path import reconstruction_frame_path
from clipped_origin_support import load_support_bundle
from first_step import save_accepted_frames
from force_state_dynamics import ImplicitDynamics, source_activation
from force_state_accuracy import POLICY
from curvature_transition import validate_curvature_transition, checkpoint_guard_sha
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
    p.add_argument('--external-audit-tag-prefix', required=True)
    p.add_argument('--same-step-q-only-guess', type=Path, help='Preserved same dt/history diagnostic q only; no old QP or barrier-stage state is imported')
    p.add_argument('--matched-passive-audit', type=Path, help='Qualified matched-time passive q-only numerical seed; active history remains its own')
    p.add_argument('--passive-first-stage-q-seed', action='store_true',
        help='Optional first-mu passive q-only guess, eligible only after its final independent audit; no physical history transfer')
    p.add_argument('--allow-cap-only-gap-search', action='store_true',
                   help='Allow isolated geometry-only SOC diagnostics for cap-rejected raw predictors; unchanged physical guards')
    p.add_argument('--reuse-source-factor-search', action='store_true',
        help='Optional one-use previous source-KKT SEARCH direction; never supplies stationarity reactions')
    p.add_argument('--barrier-policy', choices=[CONTINUATION, FIXED_POLICY], default=CONTINUATION)
    p.add_argument('--fixed-barrier-scale', type=float, default=.04)
    p.add_argument('--fixed-barrier-plan', type=Path)
    p.add_argument('--sensitivity-stage-limit', type=int, default=4,
        help='Resumable computational stage budget, not a feasibility or error claim')
    p.add_argument('--fixed-barrier-resume-candidates', nargs='+', type=Path, default=[],
        help='Exact original-history retained converged stage prefix; never accepted motion')
    a = p.parse_args()
    fixed = a.barrier_policy == FIXED_POLICY
    if fixed and (a.fixed_barrier_plan is None or not valid_scale(a.fixed_barrier_scale) or a.sensitivity_stage_limit < 2):
        p.error('Fixed mode needs a valid fixed level, immutable plan and stage budget >=2')
    fixed_plan = validate_plan(json.loads(a.fixed_barrier_plan.read_text())) if fixed else None
    if not fixed and (a.fixed_barrier_plan is not None or a.fixed_barrier_resume_candidates):
        p.error('Fixed-only provenance cannot be imported by legacy continuation')
    if fixed:
        # Optional first-mu capture is redundant here; use audited final passive q.
        a.passive_first_stage_q_seed = False
        a.barrier_scales = [a.fixed_barrier_scale]
    if not a.external_audit_tag_prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in a.external_audit_tag_prefix):
        p.error("External audit prefix requires letters, digits, hyphen or underscore")
    if a.resume_directory is None or not a.prior_interval_audits:
        p.error("Audited continuation requires prior externally qualified physical history")
    if not 0 < a.dt <= .02 or not 1 <= a.steps <= 100 or not 1 <= a.iterations <= 40 or not 0 < a.wall_seconds_per_stage <= 600:
        p.error('Bounded positive dt<=0.02s, steps<=100, iterations<=40, wall per stage<=600s required')
    if a.output.exists():
        p.error('New output directory required; preserve earlier results')
    if a.same_step_q_only_guess is not None and (a.nonphysical_warm_start is not None or a.inertia_scaled_diagnostic_guess is not None):
        p.error('Same-step q-only guess cannot be combined with another checkpoint import')
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
    if fixed:
        report.update(barrier_policy=FIXED_POLICY, fixed_barrier_scope=FIXED_SCOPE, fixed_barrier_model=FIXED_MODEL,
            fixed_barrier_starting_scale=a.fixed_barrier_scale,
            fixed_barrier_check_plan=fixed_plan, fixed_barrier_check_plan_sha256=plan_sha(fixed_plan),
            fixed_barrier_policy_source_sha256=sha(here/'fixed_barrier_policy.py'),
            barrier_qualification_source_sha256=sha(here/'barrier_reconstruction_continuation.py'),
            fixed_barrier_check_plan_path=str(a.fixed_barrier_plan.resolve()),
            fixed_barrier_check_plan_file_sha256=sha(a.fixed_barrier_plan),
            sensitivity_stage_limit=a.sensitivity_stage_limit)
    report.update(**RECONSTRUCTION_FIELDS)
    report['stationarity_accuracy_policy'] = POLICY
    report['allow_cap_only_gap_search_enabled'] = a.allow_cap_only_gap_search
    report['raw_predictor_geometry_source_sha256'] = sha(here/'raw_predictor_geometry.py')
    report['reuse_source_factor_search_enabled'] = a.reuse_source_factor_search
    report['modified_newton_search_source_sha256'] = sha(here/'modified_newton_search.py')
    if a.passive_first_stage_q_seed:
        report['stage_q_seed_source_sha256'] = {name: sha(here/name) for name in (
            'barrier_reconstruction_continuation.py', 'passive_stage_seed.py')}
    report['curvature_transition_source_sha256'] = sha(here/'curvature_transition.py')
    report['stationarity_accuracy_scope'] = 'Identical fixed fallback for both cases; original absolute target or full reaction-relative balance plus unregularized occupied-site correction'
    report['force_state_dynamics_source_sha256'] = sha(here/'force_state_dynamics.py')
    report['force_state_accuracy_source_sha256'] = sha(here/'force_state_accuracy.py')
    report['final_reconstruction_path_source_sha256'] = sha(here.parent/'contact/reconstruction-020-v1/reconstruction_020_frame_path.py')
    report['reconstruction_interval_helper_source_sha256'] = sha(here.parent/'contact/reconstruction-020-v1/continuous_closed_site_reconstruction_020.py')
    def persist():
        report['wall_seconds'] = time.perf_counter()-started
        temporary = a.output/'report.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(a.output/'report.json')
    persist()
    if prior is not None:
        if fixed:
            if a.fixed_barrier_scale != prospective_fixed_scale(prior):
                raise ValueError('Fixed level must equal the admitted prospective pair level')
            old_plan = prior.get('fixed_barrier_check_plan')
            if old_plan is not None:
                validate_plan(old_plan)
                if (prior.get('fixed_barrier_check_plan_sha256') != plan_sha(old_plan)
                        or any(check not in fixed_plan['checks'] for check in old_plan['checks'])
                        or (fixed_plan != old_plan and fixed_plan.get('predecessor_plan_sha256') != plan_sha(old_plan))):
                    raise ValueError('Fixed schedule must preserve the exact prior plan or its explicit extension')
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
        implementation_transition = validate_curvature_transition(
            prior, guard.provenance, a.guard_module, a.previous_guard_sha256,
            old_qp_import_requested=a.nonphysical_warm_start is not None)
        if implementation_transition is not None and a.nonphysical_warm_start is not None:
            raise ValueError('A changed tangent requires q-only restart; the old QP must not be imported')
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
        if fixed:
            implementations = {file.name: sha(file) for file in here.glob('*.py') if not file.name.startswith('test_')}
            implementations.update({name: sha(here.parents[1]/name) for name in ('mechanics.py', 'current_tangent.py')})
            report['fixed_stage_implementation_sha256'] = implementations
            report['fixed_stage_model_sha256'] = retained_model_sha(report, solid.provenance, implementations)
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
            if fixed:
                i = checkpoint.get('barrier_stage')
                if (not isinstance(i, int) or i < 0 or checkpoint.get('barrier_policy') != FIXED_POLICY
                        or checkpoint['barrier_scale'] != stage_scale(a.fixed_barrier_scale, i)):
                    raise ValueError('Fixed checkpoint must identify its exact selected comparison stage')
                for key in ('fixed_barrier_policy_source_sha256', 'barrier_qualification_source_sha256', 'fixed_stage_model_sha256'):
                    if checkpoint.get(key) != report[key]:
                        raise ValueError('Fixed checkpoint source differs: '+key)
            elif checkpoint['barrier_scale'] != a.barrier_scales[0]:
                raise ValueError('Internal checkpoint must resume the SAME mu')
            if checkpoint['dt_s'] != a.dt:
                raise ValueError('Internal checkpoint must resume the SAME physical timestep')
            report['nonphysical_warm_start_source'] = str(a.nonphysical_warm_start.resolve())
        dynamics = ImplicitDynamics(solid, guard, constraints=supports,
            guard_scope='coarse_physical_sites', damping_per_second=0.)
        dynamics.require_guards()
        report['status'] = 'running_coarse_barrier_transient'
        persist()
        passive_initial_guesses = {}
        if a.matched_passive_audit is not None:
            passive_initial_guesses, seed_evidence = audited_passive_endpoints(a.matched_passive_audit, report, solid.X)
            report['matched_passive_q_only_seed'] = seed_evidence
        external_hold = False
        report['external_audit_before_next_case'] = True
        report['solver_accepted_counts_scope'] = 'Provisional candidate counts until each saved case passes its independent interval audit'
        prior_audit_by_case = {json.loads(path.read_text())['case']: path.resolve() for path in a.prior_interval_audits}
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
                if a.passive_first_stage_q_seed and case == 'zygomaticus_frontalis':
                    # Explicit checkpoint workflows retain their existing priority.
                    seed_audit = a.matched_passive_audit
                    if seed_audit is None:
                        proof = report['runs'].get('passive', {}).get('external_interval_audit', {})
                        if (proof.get('qualified') is True
                                and proof.get('matching_candidate_and_immutable_proof') is True):
                            seed_audit = proof['path']
                    if any(x is not None for x in (a.same_step_q_only_guess,
                            a.nonphysical_warm_start, a.inertia_scaled_diagnostic_guess)):
                        run['first_stage_q_seed_use'] = dict(used=False,
                            reason='explicit_checkpoint_takes_precedence')
                    else:
                        try:
                            stage_guess, seed_use = audited_first_passive_stage_guess(
                                seed_audit, report, solid.X, case=case,
                                physical_time=physical_time, dt=a.dt,
                                first_scale=a.barrier_scales[0], own_previous=q,
                                guard=guard, constraints=supports)
                        except StageSeedRejected as exc:
                            stage_guess = None
                            seed_use = dict(used=False, reason='integrity_or_mapping_rejection',
                                detail=str(exc), fallback='existing independently permitted numerical guess')
                        run['first_stage_q_seed_use'] = seed_use
                        if stage_guess is not None:
                            warm = stage_guess
                            warm_source = 'audited matched first-passive-scale NONPHYSICAL q only; own active history/prediction unchanged'
                if a.same_step_q_only_guess is not None:
                    guess_metadata=json.loads(a.same_step_q_only_guess.with_suffix('.json').read_text())
                    if (sha(a.same_step_q_only_guess)!=guess_metadata['npz_sha256'] or guess_metadata.get('nonphysical_internal_iterate') is not True
                            or guess_metadata['case']!=case or guess_metadata['dt_s']!=a.dt or guess_metadata['physical_time_s']!=physical_time
                            or guess_metadata['activation']!=source_activation(physical_time,case=case,ramp_s=a.ramp_s)):
                        raise ValueError('Q-only diagnostic guess must preserve exact physical history, dt and activation')
                    for key in ('source_domain_sha256','source_phase_sha256','source_support_sha256','guard_module_sha256'):
                        expected = checkpoint_guard_sha(report, implementation_transition) if key == 'guard_module_sha256' else report[key]
                        if guess_metadata[key]!=expected:raise ValueError('Q-only diagnostic guess physics mismatch')
                    with np.load(a.same_step_q_only_guess,allow_pickle=False) as z:
                        if not np.array_equal(z['q_previous_mm'],q) or not np.array_equal(z['velocity_previous_mm_s'],v):
                            raise ValueError('Q-only diagnostic guess changed own accepted history')
                        candidate=z['nonphysical_iterate_mm'].copy()
                    if supports.error(candidate)<=1e-8 and guard.endpoint(candidate).get('feasible') is True:
                        warm=candidate;warm_source='Preserved same-step NONPHYSICAL q-only guess; barrier stages re-established, own physical history unchanged'
                partial_stage = None
                if checkpoint is not None and checkpoint['case'] == case and checkpoint['physical_time_s'] == physical_time:
                    with np.load(a.nonphysical_warm_start, allow_pickle=False) as z:
                        if not np.array_equal(z['q_previous_mm'], q) or not np.array_equal(z['velocity_previous_mm_s'], v):
                            raise ValueError('Internal checkpoint physical history differs')
                        warm = z['nonphysical_iterate_mm'].copy()
                    warm_source = 'same-step nonphysical internal checkpoint; physical history unchanged'
                    inner_history = checkpoint.get('failed_inner_report', {}).get('history', [])
                    if fixed:
                        partial_stage = dict(checkpoint, q=warm.copy())
                    elif inner_history:
                        dynamics._epigraph_qp_checkpoint = inner_history[-1].get('epigraph_qp_checkpoint')
                    if checkpoint['activation'] != source_activation(physical_time, case=case, ramp_s=a.ramp_s):
                        raise ValueError('Internal checkpoint activation differs')
                def save_checkpoint(iterate, previous, velocity, metadata):
                    path = a.output/f'{case}-step{step:04d}-NONPHYSICAL-checkpoint.npz'
                    np.savez_compressed(path, nonphysical_iterate_mm=iterate,
                        q_previous_mm=previous, velocity_previous_mm_s=velocity)
                    metadata.update(case=case, physical_time_s=physical_time, npz_sha256=sha(path),
                        **{key: report[key] for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256', 'guard_module_sha256')})
                    if fixed:
                        metadata.update(**{key: report[key] for key in (
                            'fixed_barrier_policy_source_sha256', 'barrier_qualification_source_sha256', 'fixed_stage_model_sha256')})
                    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
                    run['nonphysical_checkpoint_path'] = str(path.resolve())
                def stage_finished(stage, comparison):
                    run['pending_barrier_stages'].append(dict(scale=stage['barrier_scale'],
                        status=stage['status'], dynamic_residual_max_N=stage['dynamic_residual_max_N'],
                        wall_seconds=stage['wall_seconds'], comparison=comparison))
                    persist()
                    print(json.dumps(dict(case=case, step=step, stage='barrier_stage_finished',
                        **run['pending_barrier_stages'][-1])), flush=True)
                request = check_request(fixed_plan, times[-1] if times else 0., physical_time) if fixed else None
                retained = []
                for path in a.fixed_barrier_resume_candidates:
                    metadata = json.loads(path.with_suffix('.json').read_text())
                    if metadata.get('case') != case:
                        continue
                    origin = json.loads((path.parent/'report.json').read_text())
                    descriptors = origin.get('runs', {}).get(case, {}).get('retained_sensitivity_candidates', [])
                    if not any(item.get('path') == str(path.resolve()) and item.get('npz_sha256') == sha(path)
                            and item.get('metadata_sha256') == sha(path.with_suffix('.json')) for item in descriptors):
                        raise ValueError('Retained candidate is not bound to its producing report')
                    if (sha(path) != metadata.get('npz_sha256') or metadata.get('barrier_policy') != FIXED_POLICY
                            or metadata.get('physical_time_s') != physical_time
                            or metadata.get('nonphysical_internal_iterate') is not True
                            or metadata.get('usable_as_accepted_frame') is not False):
                        raise ValueError('Retained fixed candidate identity/history differs')
                    for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256', 'guard_module_sha256',
                                'fixed_barrier_policy_source_sha256', 'barrier_qualification_source_sha256', 'fixed_stage_model_sha256'):
                        if metadata.get(key) != report[key]:
                            raise ValueError('Retained fixed candidate source changed: '+key)
                    with np.load(path, allow_pickle=False) as z:
                        candidate = z['nonphysical_iterate_mm'].copy()
                        if (not np.array_equal(z['q_previous_mm'], q) or not np.array_equal(z['velocity_previous_mm_s'], v)
                                or array_sha(candidate) != metadata.get('candidate_q_sha256')):
                            raise ValueError('Retained candidate own-history/coordinates changed')
                    retained.append(dict(metadata, q=candidate))
                retained.sort(key=lambda r: r['barrier_stage'])
                if [r['barrier_stage'] for r in retained] != list(range(len(retained))):
                    raise ValueError('Retained candidates must be a contiguous first-stage prefix')
                def save_sensitivity_candidate(iterate, previous, velocity, metadata):
                    path = a.output/f'{case}-step{step:04d}-NONPHYSICAL-fixed-stage{metadata["barrier_stage"]:02d}.npz'
                    if path.exists() or path.with_suffix('.json').exists():
                        raise ValueError('Never overwrite a retained sensitivity candidate')
                    np.savez_compressed(path, nonphysical_iterate_mm=iterate,
                        q_previous_mm=previous, velocity_previous_mm_s=velocity)
                    metadata.update(case=case, physical_time_s=physical_time, npz_sha256=sha(path),
                        **{key: report[key] for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256',
                            'guard_module_sha256', 'fixed_barrier_policy_source_sha256', 'barrier_qualification_source_sha256', 'fixed_stage_model_sha256')})
                    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
                    run.setdefault('retained_sensitivity_candidates', []).append(dict(
                        path=str(path.resolve()), npz_sha256=sha(path), metadata_sha256=sha(path.with_suffix('.json')),
                        barrier_stage=metadata['barrier_stage'], barrier_scale=metadata['barrier_scale']))
                    persist()
                capture_callback = None
                if a.passive_first_stage_q_seed and case == 'passive':
                    try:
                        prior_seed_audit = report['prior_continuous_interval_audits'].get(case)
                        if prior_seed_audit is None:
                            raise StageSeedRejected('Capture unavailable without prior admitted passive history')
                        capture = FirstPassiveStageCapture(
                            a.output/f'{case}-step{step:04d}-NONPHYSICAL-first-mu-seed.npz',
                            report=report, source_initial=solid.X, previous=q, velocity=v,
                            previous_time=times[-1] if times else 0.,
                            target_time=physical_time, dt=a.dt, first_scale=a.barrier_scales[0],
                            prior_audit_sha256=prior_seed_audit['sha256'])
                        def capture_callback(coordinates, stage):
                            try:
                                capture(coordinates, stage)
                                if capture.descriptor is not None:
                                    run['first_stage_q_seed'] = capture.descriptor
                            except (StageSeedRejected, OSError, ValueError, KeyError, TypeError) as exc:
                                run['first_stage_q_seed_capture_error'] = str(exc)
                    except (StageSeedRejected, OSError, ValueError, KeyError, TypeError) as exc:
                        run['first_stage_q_seed_capture_error'] = str(exc)
                q, v, row = barrier_step(dynamics, q, v, dt=a.dt,
                    reuse_source_factor_search=a.reuse_source_factor_search,
                    allow_cap_only_gap_search=a.allow_cap_only_gap_search,
                    barrier_policy=a.barrier_policy, fixed_barrier_scale=a.fixed_barrier_scale,
                    sensitivity_request=request, sensitivity_stage_limit=a.sensitivity_stage_limit,
                    sensitivity_candidate_callback=save_sensitivity_candidate if fixed else None,
                    resumed_candidates=retained, resumed_partial_stage=partial_stage,
                    activation=source_activation(physical_time, case=case, ramp_s=a.ramp_s),
                    barrier_scales=a.barrier_scales, site_tolerance_mm=.005,
                    max_iterations=a.iterations, max_wall_seconds=a.wall_seconds_per_stage,
                    force_tolerance_N=a.force_tolerance_N, stage_callback=stage_finished,
                    first_initial_iterate=warm, checkpoint_callback=save_checkpoint,
                    converged_stage_q_callback=capture_callback,
                    max_operation_chunks=a.operation_chunks,
                    final_physical_path=lambda previous, candidate: reconstruction_frame_path(guard, previous, candidate),
                    progress_callback=lambda state: print(json.dumps(dict(case=case, step=step, stage='Newton', **state)), flush=True))
                row.update(**RECONSTRUCTION_FIELDS)
                row.update(physical_time_s=physical_time,
                    newton_initial_guess_source=warm_source,
                    carrier_displacement_max_mm=float(np.linalg.norm(q-solid.X, axis=1).max()))
                if fixed and row['accepted']:
                    validate_fixed_frame(row)
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
            if rows and rows[-1]['accepted'] and len(frames) > prior['runs'][case]['accepted_frames']:
                persist()
                evidence = audit_candidate_case((a.output/'report.json').resolve(), case, prior_audit_by_case[case],
                    a.external_audit_tag_prefix+'-'+case, runner=here.parent/'contact/reconstruction-020-v1/audit_reconstruction_020_intervals.py',
                    env=dict(os.environ), cwd=here)
                run['external_interval_audit'] = evidence
                persist()
                print(json.dumps(dict(case=case, stage='external_interval_audit_finished', **evidence)), flush=True)
                if not evidence['qualified']:
                    external_hold = True
                    break
            if rows and not rows[-1]['accepted'] and a.stop_after_failed_case:
                break
        report['status'] = ('external_interval_rejected_candidate_not_physical_history' if external_hold else 'bounded_barrier_attempt_finished')
    except Exception as exc:
        report.update(status='barrier_attempt_error', error_type=type(exc).__name__, blocker=str(exc))
        persist()
        raise
    persist()
    print(json.dumps(dict(status=report['status'], accepted_facial_frames=report['accepted_facial_frames'],
        wall_seconds=report['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
