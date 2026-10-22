"""Accepted physical history checks, independent of Newton warm guesses."""
import json
from pathlib import Path
import numpy as np
from source_transfer import sha
from report_storage import report_sha256
from fixed_barrier_policy import POLICY as FIXED_POLICY, validate_fixed_frame


def next_physical_time(times, dt):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(times).all():
        raise ValueError('Physical times and the new interval must be finite')
    if len(times) and (times[0] <= 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Accepted physical times must increase strictly')
    return float((times[-1] if len(times) else 0.) + dt)


def validate_accepted_history(source, frames, velocities, times, rows):
    if not (len(frames) == len(velocities) == len(times) == len(rows)):
        raise ValueError('Accepted history lengths differ')
    next_physical_time(times, 1.)
    previous, t0 = np.asarray(source), 0.
    len_validated = 0
    for q, v, t1, row in zip(frames, velocities, times, rows):
        q, v = np.asarray(q), np.asarray(v)
        if q.shape != previous.shape or v.shape != previous.shape or not (np.isfinite(q).all() and np.isfinite(v).all()):
            raise ValueError('Accepted q/v has invalid shape or values')
        dt = float(t1-t0)
        if not row['accepted'] or not np.isclose(row['dt_s'], dt, rtol=1e-12, atol=1e-14):
            raise ValueError('Saved frame dt differs from its actual cumulative interval')
        if not np.isclose(row['physical_time_s'], t1, rtol=1e-12, atol=1e-14):
            raise ValueError('Saved frame physical time differs from the NPZ')
        if not np.allclose(v, (q-previous)/dt, rtol=1e-11, atol=1e-10):
            raise ValueError('Saved velocity differs from its own physical history')
        if row.get('barrier_policy') is not None:
            own_velocity = np.zeros_like(source) if t0 == 0. else np.asarray(velocities[len_validated-1])
            validate_fixed_frame(row, previous, own_velocity, q)
        previous, t0 = q, float(t1)
        len_validated += 1


def validate_resume_interval_audits(prior, paths, cases):
    """Require the already requested external audit before further advancement."""
    audits = {}
    for path in paths:
        path = Path(path)
        record = json.loads(path.read_text())
        case = record.get('case')
        if case in audits:
            raise ValueError('Duplicate external interval audit case')
        audits[case] = (path, record)
    evidence = {}
    for case in cases:
        old = prior['runs'].get(case, {'accepted_frames': 0})
        if not old['accepted_frames']:
            continue
        if case not in audits:
            raise ValueError('Missing external continuous interval audit for '+case)
        path, record = audits[case]
        if record.get('schema') != 'continuous_closed_site_external_audit_v1' or record.get('qualified') is not True:
            raise ValueError('External continuous interval audit did not qualify')
        # Real external receipts bind their complete immutable report snapshot.
        # Keep legacy synthetic fixtures without snapshot metadata readable.
        if 'mechanics_report_path' in record or 'mechanics_report_sha256' in record:
            if report_sha256(record['mechanics_report_path']) != record['mechanics_report_sha256']:
                raise ValueError('External interval audit report snapshot changed')
        for key in ('source_domain_sha256', 'source_phase_sha256', 'source_support_sha256'):
            if record.get(key) != prior.get(key):
                raise ValueError('External interval audit source mismatch: '+key)
        if (record.get('accepted_npz_sha256') != old['accepted_npz_sha256']
                or record.get('physical_times_s') != old['actual_times_s']
                or record.get('accepted_frames_audited') != old['accepted_frames']
                or record.get('saved_velocity_history_consistent') is not True):
            raise ValueError('External interval audit does not bind the accepted physical history')
        intervals = record.get('intervals', [])
        if len(intervals) != old['accepted_frames'] or any(
                x.get('qualified') is not True
                or x.get('site_chord', {}).get('continuous_closed_owner_site_cap_checked') is not True
                or x.get('site_chord', {}).get('feasible') is not True
                or x.get('positive_J_path', {}).get('certified') is not True for x in intervals):
            raise ValueError('External interval proof is incomplete')
        evidence[case] = dict(path=str(path.resolve()), sha256=sha(path),
            accepted_npz_sha256=old['accepted_npz_sha256'],
            audited_physical_times_s=old['actual_times_s'])
    return evidence


def matched_passive_numerical_guess(case, physical_time, passive_endpoints, own_previous, guard, constraints):
    """Return only a feasible copied q guess, never physical history or velocity."""
    if case != 'zygomaticus_frontalis' or physical_time not in passive_endpoints:
        return None
    candidate=np.array(passive_endpoints[physical_time],dtype=float,copy=True)
    if candidate.shape != np.asarray(own_previous).shape or not np.isfinite(candidate).all():
        return None
    if constraints is not None and constraints.error(candidate)>1e-8:
        return None
    if guard.endpoint(candidate).get('feasible') is not True:
        return None
    return candidate


def inertia_scaled_numerical_guess(previous, velocity, failed_endpoint, old_dt, new_dt):
    """Scale acceleration correction only; preserve the new ballistic predictor."""
    previous,velocity,failed_endpoint=map(lambda x:np.asarray(x,float),(previous,velocity,failed_endpoint))
    if (not 0<new_dt<old_dt or not np.isfinite([old_dt,new_dt]).all()
            or previous.shape!=velocity.shape or previous.shape!=failed_endpoint.shape
            or not all(np.isfinite(x).all() for x in (previous,velocity,failed_endpoint))):
        raise ValueError('Finite matching history and a strictly shorter physical timestep required')
    return previous+new_dt*velocity+(new_dt/old_dt)**2*(failed_endpoint-previous-old_dt*velocity)
