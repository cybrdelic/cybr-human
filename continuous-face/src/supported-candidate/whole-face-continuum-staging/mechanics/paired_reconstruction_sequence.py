"""Bounded physical advancement using unchanged solve and external audit tools.

The common history cursor advances only after BOTH cases pass the independent
interval audit. Failed candidates remain immutable diagnostic artifacts.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from source_transfer import sha
from accepted_history import validate_resume_interval_audits
from reconstruction_seed import RECONSTRUCTION_FIELDS

CASES = ('passive', 'zygomaticus_frontalis')


def driver_decision(report, old_count, next_time):
    rows = [report.get('runs', {}).get(c, {}) for c in CASES]
    if all(r.get('accepted_frames') == old_count+1
            and r.get('actual_times_s')
            and math.isclose(r['actual_times_s'][-1], next_time, abs_tol=1e-14, rel_tol=1e-12)
            and r.get('reports') and r['reports'][-1].get('accepted') is True for r in rows):
        return 'audit_candidate_pair'
    failures=[r['reports'][-1] for r in rows if r.get('reports')
        and r['reports'][-1].get('status') == 'final_physical_path_failed']
    if failures:
        for failure in failures:
            path=failure.get('frame_path',{})
            if any(path.get(k)!=RECONSTRUCTION_FIELDS[k] for k in (
                    'reconstruction_contract','physical_endpoint_penetration_cap_mm',
                    'interframe_reconstruction_allowance_mm','certified_interframe_penetration_cap_mm')):
                return 'integrity_or_query_blocker'
            if path.get('site_chord',{}).get('offenders'):
                return 'needs_finite_reconstruction_witness'
        return 'integrity_or_query_blocker'
    return 'needs_same_stage_solver_recovery'


def audit_decision(exit_code, audit):
    if not isinstance(audit, dict) or any(audit.get(k) is not True for k in (
            'audited_snapshots_unchanged_during_audit',
            'contact_sources_unchanged_during_audit')):
        return 'integrity_or_query_blocker'
    if any(audit.get(k)!=RECONSTRUCTION_FIELDS[k] for k in (
            'reconstruction_contract','physical_endpoint_penetration_cap_mm',
            'interframe_reconstruction_allowance_mm','certified_interframe_penetration_cap_mm')):
        return 'integrity_or_query_blocker'
    if exit_code == 0 and audit.get('qualified') is True:
        return 'qualified'
    if exit_code != 2:
        return 'integrity_or_query_blocker'
    failed = [r for r in audit.get('intervals', []) if not r.get('qualified')]
    if any(r.get('site_chord',{}).get('offenders') for r in failed):
        return 'needs_finite_reconstruction_witness'
    return 'integrity_or_query_blocker'


def classification_decision(exit_code, evidence, *, case, prior_sha, candidate_sha,
                            classifier_sha, failed_audit_sha=None):
    """A true witness authorizes refinement only; it never admits a frame."""
    if not isinstance(evidence,dict) or exit_code != 0:
        return 'integrity_or_query_blocker'
    expected=dict(schema='finite_failed_chord_witness_v1', case=case,
        previous_audit_sha256=prior_sha,candidate_npz_sha256=candidate_sha,
        classifier_source_sha256=classifier_sha,classification='genuine_cap_crossing',
        genuine_cap_crossing=True,previous_q_v_unchanged=True,snapshots_unchanged=True,
        physical_force_law_changed=False,accepted_physical_frame=False)
    expected.update({key:RECONSTRUCTION_FIELDS[key] for key in ('reconstruction_contract',
        'physical_endpoint_penetration_cap_mm','interframe_reconstruction_allowance_mm','certified_interframe_penetration_cap_mm')})
    if failed_audit_sha is not None: expected['failed_audit_sha256']=failed_audit_sha
    if any(evidence.get(k)!=v for k,v in expected.items()):
        return 'integrity_or_query_blocker'
    for w in evidence.get('witnesses',[]):
        if all(w.get(k) is True for k in ('closed_owner','original_containment_valid',
                'independent_oblique_containment_valid','independent_owner_membership_matches')):
            values=[w.get(k) for k in ('finite_penetration_depth_mm','penetration_cap_mm','rounding_pad_mm')]
            if (all(isinstance(x,(int,float)) and math.isfinite(x) for x in values)
                    and values[1]==.055 and values[2]>=0 and values[0]>values[1]+values[2]
                    and isinstance(w.get('all_owner_facets_enumerated'),int)
                    and w['all_owner_facets_enumerated']>0):
                return 'refine_physical_dt'
    return 'integrity_or_query_blocker'


def qualified_dt_policy(dt, streak, maximum_dt, *, qualified):
    """Workflow choice only: three qualified pairs may grow the next dt."""
    if not qualified:
        return dt/2, 0
    streak += 1
    return (min(maximum_dt,2*dt),0) if streak>=3 else (dt,streak)


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--resume-directory', type=Path, required=True)
    p.add_argument('--prior-interval-audits', nargs=2, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-time-s', type=float, required=True)
    p.add_argument('--dt', type=float, required=True)
    p.add_argument('--maximum-dt', type=float, help='Declared original requested dt cap; default initial dt')
    p.add_argument('--minimum-retry-dt', type=float, required=True,
        help='Explicit workflow bound, not a physical feasibility claim')
    p.add_argument('--audit-tag-prefix', required=True)
    p.add_argument('--stop-file', type=Path, help='Pause after the in-flight pair is audited and admitted, without changing source files')
    a = p.parse_args()
    maximum_dt=a.dt if a.maximum_dt is None else a.maximum_dt
    if not 0 < a.minimum_retry_dt <= a.dt <= maximum_dt <= .02 or not 0 < a.target_time_s <= .1:
        p.error('Declare a bounded short sequence, dt<=20 ms and target<=100 ms')
    if not a.audit_tag_prefix or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.audit_tag_prefix):
        p.error('Audit tag prefix must contain only letters, digits, hyphen or underscore')
    a.output = a.output.resolve(); a.output.mkdir(exist_ok=False, parents=True)
    stop_file=(a.stop_file or a.output/'STOP_AT_PAIRED_BOUNDARY').resolve()
    current = a.resume_directory.resolve()
    previous = json.loads((current/'report.json').read_text())
    prior_audits = {json.loads(x.read_text())['case']: x.resolve() for x in a.prior_interval_audits}
    validate_resume_interval_audits(previous, prior_audits.values(), CASES)
    times = [previous['runs'][c]['actual_times_s'][-1] for c in CASES]
    counts = [previous['runs'][c]['accepted_frames'] for c in CASES]
    if len(set(times)) != 1 or len(set(counts)) != 1:
        raise ValueError('The sequence must start from a matched audited paired history')
    physical_time, count, dt = times[0], counts[0], a.dt
    qualified_streak=0
    if a.target_time_s <= physical_time:
        raise ValueError('Target time must exceed the current accepted time')
    driver = here/'coarse_reconstruction_driver.py'
    guard = here.parent/'contact/coarse_physical_site_contact_epigraph.py'
    audit_runner = here.parent/'contact/audit_reconstruction_intervals.py'
    classifier = here.parent/'contact/classify_failed_reconstruction_chord.py'
    # Bind the numerical core and all local contact/geometry-proof helpers,
    # not just executable entrypoints. Unused helpers are conservatively bound.
    source_paths={Path(__file__),driver,guard,audit_runner,here.parents[1]/'mechanics.py',here.parents[1]/'current_tangent.py'}
    for directory in (here,here.parent/'contact'):
        source_paths.update(x for x in directory.glob('*.py') if not x.name.startswith('test_'))
    bound_sources = {str(x.resolve()): sha(x) for x in sorted(source_paths)}
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        accepted_history_directory=str(current), accepted_physical_time_s=physical_time,
        accepted_interval_audits={c:str(x) for c,x in prior_audits.items()},
        target_time_s=a.target_time_s, fixed_barrier_scales=[.2, .04, .008],
        source_sha256=bound_sources, attempts=[],
        physical_history_policy='Commit paired cursor only after both independent interval audits; rejected candidates never become the next history',
        dt_policy='Halve only for a witnessed geometric interval failure; double after three qualified paired intervals, capped by declared original requested dt; never reinterpret solver/integrity failures as dt issues',
        initial_dt_s=a.dt,maximum_dt_s=maximum_dt,computational_retry_floor_s=a.minimum_retry_dt)
    state.update(**RECONSTRUCTION_FIELDS, safe_boundary_stop_file=str(stop_file))
    def persist():
        tmp=a.output/'sequence-state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.output/'sequence-state.json')
    def stop(status):
        state['status']=status;persist();print(json.dumps(dict(status=status,physical_time_s=physical_time)),flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env['PYTHONPATH']=str(here.parent/'contact')+os.pathsep+str(here)+os.pathsep+env.get('PYTHONPATH','')
    persist()
    while physical_time < a.target_time_s-1e-14:
        if stop_file.exists():
            stop('paused_at_requested_safe_boundary');return 0
        if any(sha(Path(path)) != digest for path,digest in bound_sources.items()):
            stop('source_integrity_blocker');return 3
        attempt_id=len(state['attempts'])+1
        trial_dt=min(dt,a.target_time_s-physical_time);next_time=physical_time+trial_dt
        output=a.output/f'interval-{count+1:04d}-attempt-{attempt_id:03d}'
        log=output.with_suffix('.log')
        attempt=dict(output=str(output), t0_s=physical_time,t1_s=next_time,dt_s=trial_dt,
            history_directory=str(current),status='solving_uncommitted_candidate_pair')
        state['attempts'].append(attempt);persist()
        command=[sys.executable,'-u',str(driver),'--guard-module',str(guard),'--output',str(output),
            '--dt',repr(trial_dt),'--steps',str(count+1),'--iterations','40','--wall-seconds-per-stage','240',
            '--operation-chunks','3','--force-tolerance-N','1e-7','--ramp-s',repr(previous['activation_ramp_s']),
            '--barrier-scales','.2','.04','.008','--cases',*CASES,'--resume-directory',str(current),
            '--prior-interval-audits',*(str(prior_audits[c]) for c in CASES),'--stop-after-failed-case',
            '--external-audit-tag-prefix',f'{a.audit_tag_prefix}-{attempt_id:03d}']
        print(json.dumps(dict(event='solve_started',**attempt)),flush=True)
        with log.open('w') as stream:
            result=subprocess.run(command,cwd=here,env=env,stdout=stream,stderr=subprocess.STDOUT)
        attempt['driver_exit_code']=result.returncode
        if result.returncode or not (output/'report.json').exists():
            attempt['status']='driver_exception';stop('needs_solver_or_input_recovery');return 3
        candidate=json.loads((output/'report.json').read_text())
        decision=driver_decision(candidate,count,next_time)
        attempt['solver_decision']=decision
        new_audits={}; failure_case=None; failed_audit=None
        for case in CASES:
            run=candidate.get('runs',{}).get(case,{})
            proof=run.get('external_interval_audit')
            if proof is None:
                if run.get('reports') and run['reports'][-1].get('status')=='final_physical_path_failed':
                    failure_case=case
                break
            audit_path=Path(proof['path'])
            audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
            action=audit_decision(proof['exit_code'],audit)
            if not audit_path.is_file() or proof.get('sha256') != sha(audit_path) or proof.get('matching_candidate_and_immutable_proof') is not True:
                action='integrity_or_query_blocker'
            attempt.setdefault('audits',{})[case]=dict(path=str(audit_path),exit_code=proof['exit_code'],decision=action)
            persist()
            if action!='qualified':
                decision=action;failure_case=case;failed_audit=audit_path;break
            new_audits[case]=audit_path
        if len(new_audits)==len(CASES) and decision=='audit_candidate_pair':
            # Only BOTH immutable external proofs admit the paired physical cursor.
            if any(not Path(path).is_file() or sha(Path(path)) != digest for path,digest in bound_sources.items()):
                attempt['status']='source_integrity_blocker';stop('source_integrity_blocker');return 3
            validate_resume_interval_audits(candidate,new_audits.values(),CASES)
            current,previous,prior_audits=output,candidate,new_audits
            physical_time,count=next_time,count+1
            dt,qualified_streak=qualified_dt_policy(trial_dt,qualified_streak,maximum_dt,qualified=True)
            attempt['status']='paired_interval_qualified'
            state.update(accepted_history_directory=str(current),accepted_physical_time_s=physical_time,
                accepted_interval_audits={c:str(x) for c,x in prior_audits.items()},
                next_dt_s=dt,qualified_pair_streak=qualified_streak)
            persist();print(json.dumps(dict(event='paired_interval_qualified',physical_time_s=physical_time,
                output=str(current),audits=state['accepted_interval_audits'])),flush=True)
            continue
        if failure_case is not None and decision!='refine_physical_dt':
            tag=f'{a.audit_tag_prefix}-{attempt_id:03d}-{failure_case}-witness'
            command=[sys.executable,'-u',str(classifier),'--mechanics-report',str(output/'report.json'),
                '--case',failure_case,'--previous-audit',str(prior_audits[failure_case]),'--output-tag',tag]
            if failed_audit is not None: command+=['--failed-audit',str(failed_audit)]
            with (a.output/f'{tag}.log').open('w') as stream:
                classification=subprocess.run(command,cwd=here,env=env,stdout=stream,stderr=subprocess.STDOUT)
            classification_path=here.parent/'contact/temporal-audits'/f'{tag}-classification.json'
            evidence=json.loads(classification_path.read_text()) if classification_path.exists() else None
            run=candidate['runs'][failure_case]
            candidate_sha=(run.get('accepted_npz_sha256') if failed_audit is not None
                else sha(Path(run['nonphysical_checkpoint_path'])))
            decision=classification_decision(classification.returncode,evidence,case=failure_case,
                prior_sha=sha(prior_audits[failure_case]),candidate_sha=candidate_sha,
                classifier_sha=bound_sources[str(classifier.resolve())],
                failed_audit_sha=sha(failed_audit) if failed_audit else None)
            attempt['finite_witness_classification']=dict(path=str(classification_path),
                exit_code=classification.returncode,decision=decision)
        if any(not Path(path).is_file() or sha(Path(path)) != digest for path,digest in bound_sources.items()):
            stop('source_integrity_blocker');return 3
        attempt.update(status='rejected_candidate_not_physical_history',rejection=decision,
            physical_history_rolled_back_to=str(current),candidate_is_diagnostic_only=True)
        persist()
        if decision!='refine_physical_dt':
            stop(decision);return 3
        dt,qualified_streak=qualified_dt_policy(trial_dt,qualified_streak,maximum_dt,qualified=False)
        if dt<a.minimum_retry_dt-1e-14:
            stop('smaller_dt_needed_beyond_declared_workflow_bound_not_physical_infeasibility');return 2
        print(json.dumps(dict(event='retry_shorter_actual_interval',dt_s=dt,unchanged_history=str(current))),flush=True)
    stop('short_paired_sequence_qualified');return 0


if __name__=='__main__':
    sys.exit(main())

