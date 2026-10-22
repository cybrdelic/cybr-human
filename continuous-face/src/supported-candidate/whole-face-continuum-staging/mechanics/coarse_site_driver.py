"""Actual finite-resolution physical-site transient, never exact boundary CAD physics.

Separate opt-in guard scope. Same physical mass and exact material equations.
Only converged steps are persisted; passive/active histories share the source
initial state, gravity, time steps, damping and finite native contact model.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from clipped_origin_support import load_support_bundle
from first_step import save_accepted_frames
from implicit_dynamics import ImplicitDynamics, MissingGeometryCertification, source_activation
from source_transfer import load_geometry_with_source_fibres, sha


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--guard-module', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dt', type=float, default=.01)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--wall-seconds-per-step', type=float, default=90.)
    p.add_argument('--force-tolerance-N', type=float, default=1e-7)
    p.add_argument('--ramp-s', type=float, default=.1)
    p.add_argument('--cases', nargs='+', default=['passive', 'zygomaticus_frontalis'])
    p.add_argument('--resume-directory', type=Path, help='Continue hash-bound accepted states, preserving their actual times and velocities')
    a = p.parse_args()
    if not .0001 <= a.dt <= .02 or not 1 <= a.steps <= 10 or not 1 <= a.iterations <= 16 or not 0 < a.wall_seconds_per_step <= 180 or not .02 <= a.ramp_s <= .2:
        p.error('bounded coarse branch:dt0.0001-0.02s,1-10steps,1-16iterations,budget<=180s,ramp0.02-0.2s')
    if a.output.exists():
        p.error('New output directory required; preserve prior actual states and reports')
    a.output.mkdir(parents=True)
    started = time.perf_counter()
    domain = here.parent/'geometry/whole-face-continuum-final.npz'
    phases = here.parent/'geometry/whole-face-muscle-phases-final.npz'
    solid = load_geometry_with_source_fibres(domain, phases)
    supports, support_report = load_support_bundle(solid, here/'coarse-physical-site-v1/supports')
    prior = None
    if a.resume_directory is not None:
        prior = json.loads((a.resume_directory/'report.json').read_text())
        if (prior.get('source_domain_sha256') != sha(domain) or prior.get('source_phase_sha256') != sha(phases)
                or prior.get('physical_dt_s') != a.dt or prior.get('activation_ramp_s') != a.ramp_s
                or prior.get('contact_mode') != 'coarse_physical_sites'):
            raise ValueError('Continuation model, physical time step or activation differs')
    report = dict(status='preparing_coarse_contact', guard_scope='coarse_physical_sites',
        contact_mode='coarse_physical_sites', global_surface_collision_certified=False, site_resolution_mm=1.,
        model_resolution='6-L0 P1 carrier, positive1-L0 occupied voxels, physical-site native contact; no exact whole-boundary collision claim',
        source_domain_sha256=sha(domain), source_phase_sha256=sha(phases),
        physical_dt_s=a.dt, requested_steps_per_case=a.steps, activation_ramp_s=a.ramp_s,
        damping_per_second=0., gravity_mm_s2=[0., -9810., 0.],
        physical_mass_kg=float(solid.mass.sum()*1000), initial_velocity='zero',
        initial_state='registered source, non-equilibrated transient settling',
        length_scale='1L0=1mm is an explicit engineering assumption, not patient calibration',
        no_artificial_mass_or_stiffness=True, source_origin_support= support_report,
        accepted_facial_frames=0, exact_boundary_certified_frames=0, runs={})
    if prior is not None:
        report['resumed_report_path'] = str((a.resume_directory/'report.json').resolve())
        report['resumed_report_sha256'] = sha(a.resume_directory/'report.json')
    def persist():
        report['wall_seconds'] = time.perf_counter()-started
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    persist()
    try:
        spec = importlib.util.spec_from_file_location('supplied_coarse_physical_site_guard', a.guard_module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        guard = module.create_guard(solid, supports)
        report.update(guard_module=str(a.guard_module.resolve()), guard_module_sha256=sha(a.guard_module),
            coarse_contact_provenance=getattr(guard, 'provenance', {}),
            guard_capabilities=getattr(guard, 'certification', {}))
        solver = ImplicitDynamics(solid, guard, constraints=supports, guard_scope='coarse_physical_sites', damping_per_second=0.)
        solver.require_guards()
        report['status'] = 'running_coarse_physical_transient'
        persist()
        print('Coarse guard ready; beginning actual physical transient.', flush=True)
        for case in a.cases:
            q, v = solid.X.copy(), np.zeros_like(solid.X)
            frames, velocities, times, rows = [], [], [], []
            if prior is not None:
                old = prior['runs'][case]
                saved = Path(old['accepted_npz_path'])
                if sha(saved) != old['accepted_npz_sha256']:
                    raise ValueError('Continuation state SHA mismatch')
                with np.load(saved, allow_pickle=False) as z:
                    if not np.array_equal(z['source_initial_positions_mm'], solid.X):
                        raise ValueError('Continuation uses a different source reference')
                    frames = [x.copy() for x in z['positions_mm']]
                    velocities = [x.copy() for x in z['velocities_mm_s']]
                    times = z['physical_times_s'].tolist()
                rows = [r for r in old['reports'] if r['accepted']]
                if len(rows) != len(frames) or times != old['actual_times_s'] or not frames:
                    raise ValueError('Continuation lacks matching accepted physical evidence')
                q, v = frames[-1].copy(), velocities[-1].copy()
                report['accepted_facial_frames'] += len(frames)
            report['runs'][case] = dict(accepted_frames=0, actual_times_s=[], reports=rows)
            if frames:
                report['runs'][case].update(accepted_frames=len(frames), actual_times_s=times,
                    **save_accepted_frames(a.output/f'{case}-accepted.npz', positions=frames, velocities=velocities, times=times, solid=solid))
                persist()
            for step in range(len(frames)+1, a.steps+1):
                physical_time = step*a.dt
                print(json.dumps(dict(case=case, step=step, physical_time_s=physical_time, stage='step_start')), flush=True)
                q, v, row = solver.step(q, v, dt=a.dt,
                    activation=source_activation(physical_time, case=case, ramp_s=a.ramp_s),
                    max_iterations=a.iterations, force_tolerance_N=a.force_tolerance_N,
                    max_wall_seconds=a.wall_seconds_per_step,
                    progress_callback=lambda state: print(json.dumps(dict(case=case, step=step, stage='Newton', **state)), flush=True))
                row['physical_time_s'] = physical_time
                row['carrier_displacement_max_mm'] = float(np.linalg.norm(q-solid.X, axis=1).max())
                rows.append(row)
                if row['accepted']:
                    frames.append(q.copy()); velocities.append(v.copy()); times.append(physical_time)
                    report['accepted_facial_frames'] += 1
                    run = report['runs'][case]
                    run.update(accepted_frames=len(frames), actual_times_s=times,
                        **save_accepted_frames(a.output/f'{case}-accepted.npz', positions=frames, velocities=velocities, times=times, solid=solid))
                persist()
                print(json.dumps(dict(case=case, step=step, status=row['status'], accepted=row['accepted'], wall_seconds=row['wall_seconds'], iterations=len(row['history'])-1,
                    dynamic_residual_N=row['dynamic_residual_max_N'], physical_force_N=row['physical_force_max_N'], minimum_J=row['minimum_J'], carrier_displacement_max_mm=row['carrier_displacement_max_mm'])), flush=True)
                if not row['accepted']:
                    break
        report['status'] = 'bounded_coarse_attempt_finished'
    except MissingGeometryCertification as exc:
        report.update(status='blocked_coarse_motion', blocker=str(exc))
    except Exception as exc:
        report.update(status='coarse_attempt_error', error_type=type(exc).__name__, blocker=str(exc))
        persist()
        raise
    persist()
    print(json.dumps(dict(status=report['status'], accepted_facial_frames=report['accepted_facial_frames'], wall_seconds=report['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
