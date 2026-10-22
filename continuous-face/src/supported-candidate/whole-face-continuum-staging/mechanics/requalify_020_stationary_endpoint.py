"""Recheck a preserved stationary endpoint under a new chord-only contract.

No dynamics are rerun and no original report is modified. The new case remains
unpaired until its independently frozen interval audit passes.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
from source_transfer import sha, load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle
from accepted_history import validate_resume_interval_audits, validate_accepted_history
from first_step import save_accepted_frames
from reconstruction_020_seed import RECONSTRUCTION_FIELDS


def requalified_row(original, path):
    if original.get('status') != 'final_physical_path_failed' or original.get('accepted') is not False:
        raise ValueError('Only an unchanged final-path-rejected stationary endpoint is reusable')
    for key in ('final_inner_stationarity_converged', 'barrier_continuation_qualified',
                'lower_mu_comparison_passed', 'strict_positive_slack',
                'complementarity_bound_satisfied', 'contact_generalized_stationarity_qualified'):
        if original.get(key) is not True:
            raise ValueError('Missing original numerical qualification: '+key)
    for key in ('reconstruction_contract', 'physical_endpoint_penetration_cap_mm',
                'interframe_reconstruction_allowance_mm', 'certified_interframe_penetration_cap_mm'):
        if path.get(key) != RECONSTRUCTION_FIELDS[key]:
            raise ValueError('New path uses a different reconstruction contract: '+key)
    if path.get('feasible') is not True or path.get('coarse_whole_segment_checked') is not True:
        raise ValueError('The full new-limit path must independently pass')
    row=copy.deepcopy(original)
    row.update(RECONSTRUCTION_FIELDS)
    row.update(status='converged', accepted=True, frame_path=path,
        final_physical_path_qualified=True, physical_frame_path_deferred=False,
        reconstructed_from_unchanged_stationary_candidate=True)
    row['barrier_continuation']['committed_physical_time_steps']=1
    return row


def bind_stationary_metadata(original, metadata):
    if metadata.get('dt_s') != original.get('dt_s') or metadata.get('activation') != original.get('activation'):
        raise ValueError('Checkpoint dt or activation differs from the stationary equations')
    if metadata.get('barrier_scale') != original.get('barrier_continuation',{}).get('accepted_barrier_scale'):
        raise ValueError('Checkpoint mu differs from the stationary parameter state')


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser()
    p.add_argument('--mechanics-report',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--previous-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('A new output directory is required')
    old=json.loads(a.mechanics_report.read_text())
    metadata_path=a.checkpoint.with_suffix('.json');meta=json.loads(metadata_path.read_text())
    if meta.get('npz_sha256')!=sha(a.checkpoint) or meta.get('final_parameter_stationary_but_physical_path_failed') is not True:
        raise ValueError('Checkpoint is not the exact preserved stationary candidate')
    case=meta['case'];run=old['runs'][case]
    original=run['reports'][-1]
    for key,value in meta['failed_inner_report'].items():
        if original.get(key)!=value:raise ValueError('Saved numerical report mismatch: '+key)
    bind_stationary_metadata(original,meta)
    validate_resume_interval_audits(old,[a.previous_audit],[case])
    for key in ('source_domain_sha256','source_phase_sha256','source_support_sha256','guard_module_sha256'):
        if old[key]!=meta[key]:raise ValueError('Checkpoint model mismatch: '+key)
    prefix=Path(run['accepted_npz_path'])
    if sha(prefix)!=run['accepted_npz_sha256']:raise ValueError('Accepted prefix changed')
    with np.load(prefix,allow_pickle=False) as z:
        positions=z['positions_mm'].copy();velocities=z['velocities_mm_s'].copy();times=z['physical_times_s'].tolist();X=z['source_initial_positions_mm'].copy()
    with np.load(a.checkpoint,allow_pickle=False) as z:
        candidate=z['nonphysical_iterate_mm'].copy();previous=z['q_previous_mm'];velocity=z['velocity_previous_mm_s']
        if not np.array_equal(previous,positions[-1]) or not np.array_equal(velocity,velocities[-1]):
            raise ValueError('Checkpoint physical history differs from the admitted prefix')
    dt=meta['dt_s'];physical_time=meta['physical_time_s']
    if not np.isclose(times[-1]+dt,physical_time,rtol=0,atol=1e-14):raise ValueError('Physical time mismatch')
    domain=here.parent/'geometry/whole-face-continuum-final.npz';phases=here.parent/'geometry/whole-face-muscle-phases-final.npz'
    support_dir=here/'coarse-physical-site-v1/supports';guard_path=here.parent/'contact/coarse_physical_site_contact_epigraph.py'
    for key,path in [('source_domain_sha256',domain),('source_phase_sha256',phases),('source_support_sha256',support_dir/'clipped-origin-support.json'),('guard_module_sha256',guard_path)]:
        if sha(path)!=old[key]:raise ValueError('Original mechanics input changed: '+key)
    for name,digest in old['source_support_bundle_sha256'].items():
        if sha(support_dir/name)!=digest:raise ValueError('Source support bundle changed')
    solid=load_geometry_with_source_fibres(domain,phases)
    if not np.array_equal(X,solid.X):raise ValueError('Source geometry changed')
    supports,_=load_support_bundle(solid,support_dir)
    spec=importlib.util.spec_from_file_location('stationary_requalification_guard',guard_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    guard=module.create_guard(solid,supports);guard.set_barrier_scale(meta['barrier_scale'])
    for key,value in old['coarse_contact_provenance'].items():
        if key.endswith('sha256') and guard.provenance.get(key)!=value:raise ValueError('Physical contact implementation changed: '+key)
    from reconstruction_020_frame_path import reconstruction_frame_path
    path=reconstruction_frame_path(guard,positions[-1],candidate)
    row=requalified_row(original,path)
    row['carrier_displacement_max_mm']=float(np.linalg.norm(candidate-X,axis=1).max())
    positions=np.concatenate((positions,candidate[None]),axis=0)
    velocities=np.concatenate((velocities,((candidate-positions[-2])/dt)[None]),axis=0)
    times.append(physical_time)
    report=copy.deepcopy(old);report.update(RECONSTRUCTION_FIELDS)
    new_run=copy.deepcopy(run);new_run['reports']=[*run['reports'][:-1],row]
    new_run.update(accepted_frames=len(times),actual_times_s=times)
    validate_accepted_history(X,positions,velocities,times,new_run['reports'])
    a.output.mkdir(parents=True)
    new_run.update(save_accepted_frames(a.output/f'{case}-accepted.npz',positions=positions,velocities=velocities,times=times,solid=solid))
    report.update(runs={case:new_run},accepted_facial_frames=len(times),status='requalified_stationary_candidate_pending_independent_interval_audit',
        final_reconstruction_path_source_sha256=sha(here.parent/'contact/reconstruction-020-v1/reconstruction_020_frame_path.py'),
        reconstruction_interval_helper_source_sha256=sha(here.parent/'contact/reconstruction-020-v1/continuous_closed_site_reconstruction_020.py'),
        reconstruction_policy_transition=dict(old_contract=old['reconstruction_contract'],new_contract=RECONSTRUCTION_FIELDS['reconstruction_contract'],
            original_failed_report_path=str(a.mechanics_report.resolve()),original_failed_report_sha256=sha(a.mechanics_report),
            stationary_checkpoint_path=str(a.checkpoint.resolve()),stationary_checkpoint_sha256=sha(a.checkpoint),
            checkpoint_metadata_sha256=sha(metadata_path),unchanged_stationary_endpoint_coordinates=True,
            unchanged_physical_equations_reactions_and_history=True,velocity_recomputed_from_original_history_once=True,
            reused_numerical_stage_reports_without_resolve=True,old_rejection_preserved=True,
            prior_admitted_audit_path=str(a.previous_audit.resolve()),prior_admitted_audit_sha256=sha(a.previous_audit),
            independent_new_contract_audit_still_required=True,paired_history_admitted=False))
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report_path=str((a.output/'report.json').resolve()),accepted_npz_path=new_run['accepted_npz_path'],physical_time_s=physical_time,paired_history_admitted=False)))
if __name__=='__main__':main()
