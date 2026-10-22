"""Bounded read-only witnesses for the fixed 0.07 mm reconstruction cap.

Reuses the frozen complete finite query/oblique-ray witness implementation.
Physical endpoint contact cap and forces remain unchanged at 0.05 mm.

Never solves dynamics or accepts a frame. A genuine witness supports timestep
refinement; it does not repair an unrelated invalid query or certify a path.
"""
from pathlib import Path
import sys
CONTACT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(CONTACT))
import argparse,hashlib,io,json,time
import numpy as np
from coarse_physical_site_contact_epigraph import create_guard
from coarse_physical_site_contact import sha
from coarse_rigid_query import _bins
from source_transfer import load_geometry_with_source_fibres
from report_storage import read_report_bytes, write_report_snapshot, report_sha256, report_raw_sha256
from clipped_origin_support import load_support_bundle

HERE=Path(__file__).resolve().parent


from classify_failed_site_chord import finite_cap_witnesses


def reconstruction_offenders(chord):
    """Only the hash-bound fixed reconstruction limit, never an arbitrary cap."""
    if (chord.get('reconstruction_contract')!='fixed_0p02mm_interframe_allowance_v1'
            or chord.get('reconstruction_contract_schema')!='fixed_reconstruction_accuracy_v2'
            or chord.get('physical_endpoint_penetration_cap_mm')!=.05
            or chord.get('interframe_reconstruction_allowance_mm')!=.02
            or chord.get('certified_interframe_penetration_cap_mm')!=.07
            or chord.get('penetration_cap_mm')!=.07
            or chord.get('source_sha256')!=sha(HERE/'continuous_closed_site_reconstruction_020.py')):
        raise ValueError('A matching fixed reconstruction-limit certificate is required')
    return .07,chord.get('offenders',[])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mechanics-report',type=Path,required=True)
    p.add_argument('--case',choices=['passive','zygomaticus_frontalis'],required=True)
    p.add_argument('--previous-audit',type=Path,required=True)
    p.add_argument('--output-tag',required=True)
    p.add_argument('--failed-audit',type=Path,help='Required for an external interval-audit failure; never repeat that audit')
    a=p.parse_args()
    if not a.output_tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.output_tag):p.error('Simple new output tag required')
    out=HERE/'temporal-audits';out.mkdir(exist_ok=True)
    result_path=out/(a.output_tag+'-classification.json')
    if result_path.exists():p.error('Preserve earlier classification; use a new tag')
    storage_source=Path(sys.modules['report_storage'].__file__)
    storage_source_sha=sha(storage_source)
    started=time.perf_counter();raw=read_report_bytes(a.mechanics_report);report=json.loads(raw);run=report['runs'][a.case]
    prior_raw=a.previous_audit.read_bytes();prior=json.loads(prior_raw)
    if prior.get('qualified') is not True or prior.get('case')!=a.case:raise ValueError('Matching qualified prior interval audit required')
    if sha(prior['accepted_npz_snapshot_path'])!=prior['accepted_npz_sha256'] or report_sha256(prior['mechanics_report_path'])!=prior['mechanics_report_sha256']:
        raise ValueError('Prior accepted audit snapshots changed')
    with np.load(prior['accepted_npz_snapshot_path']) as z:
        q0=z['positions_mm'][-1];v0=z['velocities_mm_s'][-1];t0=float(z['physical_times_s'][-1])
    rows=run['reports'];candidate_meta=None;metadata_raw=None;failed=None
    if a.failed_audit:
        failed_raw=a.failed_audit.read_bytes();failed=json.loads(failed_raw)
        if failed.get('qualified') is not False or failed.get('case')!=a.case:raise ValueError('Matching failed external audit required')
        if not failed.get('contact_sources_unchanged_during_audit') or not failed.get('audited_snapshots_unchanged_during_audit'):
            raise ValueError('Failed external audit has an integrity problem, not a geometric classification target')
        if report_sha256(failed['mechanics_report_path'])!=failed['mechanics_report_sha256']:
            raise ValueError('Failed external audit report snapshot changed')
        source=Path(failed['accepted_npz_snapshot_path']);data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=failed['accepted_npz_sha256'] or run['accepted_npz_sha256']!=failed['accepted_npz_sha256']:
            raise ValueError('Failed candidate bytes do not match the mechanics result')
        with np.load(io.BytesIO(data)) as z:
            if len(z['positions_mm'])<2 or not np.array_equal(z['positions_mm'][-2],q0) or not np.array_equal(z['velocities_mm_s'][-2],v0):
                raise ValueError('Failed candidate does not start from the prior accepted q/v')
            q1=z['positions_mm'][-1];t1=float(z['physical_times_s'][-1])
        interval=failed['intervals'][-1]
        if interval['qualified'] or interval['t0_s']!=t0 or interval['t1_s']!=t1:raise ValueError('Failed external interval/history mismatch')
    else:
        row=rows[-1]
        if row.get('status')!='final_physical_path_failed' or row.get('accepted') is not False:
            raise ValueError('Only a stationary final physical-path failure is eligible')
        source=Path(run['nonphysical_checkpoint_path']);data=source.read_bytes()
        metadata_raw=source.with_suffix('.json').read_bytes();candidate_meta=json.loads(metadata_raw)
        if (candidate_meta.get('final_parameter_stationary_but_physical_path_failed') is not True
                or candidate_meta.get('usable_as_accepted_frame') is not False
                or candidate_meta['npz_sha256']!=hashlib.sha256(data).hexdigest()
                or candidate_meta['case']!=a.case or candidate_meta['physical_time_s']!=row['physical_time_s']):
            raise ValueError('Checkpoint is not the hash-bound stationary failed candidate')
        with np.load(io.BytesIO(data)) as z:
            if not np.array_equal(z['q_previous_mm'],q0) or not np.array_equal(z['velocity_previous_mm_s'],v0):
                raise ValueError('Checkpoint changed prior accepted physical q/v')
            q1=z['nonphysical_iterate_mm']
        t1=float(row['physical_time_s'])
        if not np.isclose(t1-t0,row['dt_s'],rtol=1e-12,atol=1e-15):raise ValueError('Checkpoint physical time mismatch')
    if q1.shape!=q0.shape or not np.isfinite(q1).all():raise ValueError('Finite candidate coordinates required')
    report_copy=out/(a.output_tag+'-mechanics-report.json');state_copy=out/(a.output_tag+'-candidate.npz')
    if report_copy.exists() or state_copy.exists():raise ValueError('New output tag required')
    report_snapshot=write_report_snapshot(report_copy,raw);state_copy.write_bytes(data)
    if metadata_raw is not None:
        metadata_copy=out/(a.output_tag+'-checkpoint-metadata.json');metadata_copy.write_bytes(metadata_raw)
    solid=load_geometry_with_source_fibres(CONTACT.parent/'geometry/whole-face-continuum-final.npz',CONTACT.parent/'geometry/whole-face-muscle-phases-final.npz')
    supports,_=load_support_bundle(solid,CONTACT.parent/'mechanics/coarse-physical-site-v1/supports')
    guard=create_guard(solid,supports);native=guard.rigid.native
    expected=report['coarse_contact_provenance']
    for key,value in guard.provenance.items():
        if key.endswith('_sha256') and expected.get(key)!=value:raise ValueError('Changed contact/query source: '+key)
    for key,path in [('source_domain_sha256',CONTACT.parent/'geometry/whole-face-continuum-final.npz'),
                     ('source_phase_sha256',CONTACT.parent/'geometry/whole-face-muscle-phases-final.npz'),
                     ('source_support_sha256',CONTACT.parent/'mechanics/coarse-physical-site-v1/supports/clipped-origin-support.json')]:
        if report[key]!=prior[key]:raise ValueError('Changed prior physical source: '+key)
        if sha(path)!=report[key]:raise ValueError('Actual physical source differs from report: '+key)
        if failed is not None and failed[key]!=report[key]:raise ValueError('Failed audit physical source mismatch: '+key)
    for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
        if expected[key]!=prior['physical_query_data_sha256'][key]:raise ValueError('Changed prior query data: '+key)
    chord=row['frame_path'].get('site_chord',{}) if failed is None else interval['site_chord']
    certificate_cap,offenders=reconstruction_offenders(chord)
    ids=np.array([x['site'] for x in offenders],int);alphas=np.array([x['alpha'] for x in offenders],float)
    if np.any(ids<0) or np.any(ids>=guard.C.shape[0]):raise ValueError('Unknown physical site witness')
    P0,P1=guard.C[ids]@q0,guard.C[ids]@q1;points=P0+alphas[:,None]*(P1-P0)
    if any(not np.allclose(point,offender['point_mm'],rtol=0,atol=1e-11) for point,offender in zip(points,offenders)):
        raise ValueError('Reconstruction offender differs from its actual affine chord')
    evidence=finite_cap_witnesses(native,points,ids,alphas,certificate_cap)
    for witness in evidence['witnesses']:
        site,owner=witness['site'],witness['owner']
        endpoints=np.r_[guard.C[site]@q0,guard.C[site]@q1]
        inside,valid=native.containment(endpoints);slot=np.searchsorted(native.owner_ids,owner)
        faces=native.owners[owner]['ids'];history=[]
        for j,point in enumerate(endpoints):
            closest,d2=native._finite_witness(np.broadcast_to(point,(len(faces),3)),faces)
            win=np.lexsort((faces,d2))[0];distance=float(np.sqrt(d2[win]))
            history.append(dict(alpha=float(j),point_mm=point.tolist(),containment_valid=bool(valid[j]),
                inside=bool(inside[j,slot]),signed_gap_mm=(-distance if inside[j,slot] else distance),
                closest_filtered_face=int(faces[win]),all_owner_facets_enumerated=len(faces)))
        witness['endpoint_history']=history
        witness['physical_site_chord_displacement_mm']=float(np.linalg.norm(endpoints[1]-endpoints[0]))
    evidence.update(schema='finite_failed_chord_witness_v1',case=a.case,physical_interval_s=[t0,t1],
        reconstruction_contract='fixed_0p02mm_interframe_allowance_v1',
        reconstruction_contract_schema='fixed_reconstruction_accuracy_v2',trajectory_error_bound_proven=False,
        physical_endpoint_penetration_cap_mm=.05,interframe_reconstruction_allowance_mm=.02,
        certified_interframe_penetration_cap_mm=certificate_cap,
        finite_witness_implementation_sha256=sha(CONTACT/'classify_failed_site_chord.py'),
        continuous_reconstruction_source_sha256=sha(HERE/'continuous_closed_site_reconstruction_020.py'),
        candidate_npz_snapshot_path=str(state_copy),candidate_npz_sha256=sha(state_copy),
        mechanics_report_snapshot_path=str(report_copy),mechanics_report_sha256=sha(report_copy),
        mechanics_report_raw_sha256=report_snapshot.raw_sha256,
        mechanics_report_storage=report_snapshot.identity(),
        report_storage_source_sha256=storage_source_sha,
        previous_audit_path=str(a.previous_audit.resolve()),previous_audit_sha256=hashlib.sha256(prior_raw).hexdigest(),
        previous_accepted_npz_sha256=prior['accepted_npz_sha256'],previous_q_v_unchanged=True,
        failed_sites_available=len(ids),uninspected_sites=max(0,len(ids)-evidence['checked_sites']),
        classifier_source_sha256=sha(Path(__file__)),native_source_sha256=guard.provenance['native_source_sha256'],
        rigid_query_source_sha256=guard.provenance['rigid_query_source_sha256'],
        physical_force_law_changed=False,accepted_physical_frame=False,global_surface_collision_certified=False,
        wall_seconds=time.perf_counter()-started)
    if a.failed_audit:evidence['failed_audit_sha256']=hashlib.sha256(failed_raw).hexdigest()
    if candidate_meta:
        evidence['checkpoint_metadata_snapshot_path']=str(metadata_copy)
        evidence['checkpoint_metadata_sha256']=hashlib.sha256(metadata_raw).hexdigest()
    evidence['report_storage_source_unchanged']=sha(storage_source)==storage_source_sha
    evidence['snapshots_unchanged']=evidence['report_storage_source_unchanged'] and sha(state_copy)==hashlib.sha256(data).hexdigest() and report_sha256(report_copy)==report_snapshot.file_sha256 and report_raw_sha256(report_copy)==hashlib.sha256(raw).hexdigest()
    if not evidence['snapshots_unchanged']:raise ValueError('Classification snapshots changed')
    result_path.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(classification_path=str(result_path),classification_sha256=sha(result_path),classification=evidence['classification'],witnesses=len(evidence['witnesses']),wall_seconds=evidence['wall_seconds'])),flush=True)
    if not evidence['genuine_cap_crossing']:raise SystemExit(2)


if __name__=='__main__':main()
