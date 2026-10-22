"""Bounded read-only finite cap witnesses for a failed physical site chord.

Never solves dynamics or accepts a frame. A genuine witness supports timestep
refinement; it does not repair an unrelated invalid query or certify a path.
"""
from pathlib import Path
import argparse,hashlib,io,json,time
import numpy as np
from coarse_physical_site_contact_epigraph import create_guard
from coarse_physical_site_contact import sha
from coarse_rigid_query import _bins
from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle

HERE=Path(__file__).resolve().parent


def finite_cap_witnesses(native,points,site_ids,alphas,cap,*,max_points=32):
    points=np.asarray(points,float);site_ids=np.asarray(site_ids,int);alphas=np.asarray(alphas,float)
    if not np.isfinite(cap) or cap<=0 or not isinstance(max_points,int) or max_points<1:
        raise ValueError('Positive finite cap and positive bounded witness count required')
    if (points.shape!=(len(site_ids),3) or alphas.shape!=(len(site_ids),)
            or not np.isfinite(points).all() or not np.isfinite(alphas).all() or np.any((alphas<0)|(alphas>1))):
        raise ValueError('Finite, matching site/alpha/witness points required')
    count=min(len(points),max_points);points,site_ids,alphas=points[:count],site_ids[:count],alphas[:count]
    if not count:return dict(classification='inconclusive',genuine_cap_crossing=False,witnesses=[],checked_sites=0,unclassified_checked_sites=[],finite_distance_evaluations=0)
    inside,valid=native.containment(points)
    direction=np.array([.573,.231,1.]);direction/=np.linalg.norm(direction)
    e1=np.cross(direction,[0.,1.,0.]);e1/=np.linalg.norm(e1)
    matrix=np.column_stack((e1,np.cross(direction,e1),direction))
    tri=native.ray_triangles@matrix;bounds=[]
    for owner in native.owner_ids:
        t=native.triangles[native.owners[int(owner)]['ids']]@matrix
        bounds.append((t.min((0,1)),t.max((0,1))))
    oblique,oblique_valid=native._ray(points@matrix,tri,_bins(tri,2),bounds=bounds)
    witnesses=[];unresolved=[];finite_distance_evaluations=0
    scale=np.max(abs(native.vertices),initial=0.)
    for j,point in enumerate(points):
        signs_ok=bool(valid[j] and oblique_valid[j] and np.array_equal(inside[j],oblique[j]))
        found=False
        if signs_ok:
            for owner in native.owner_ids[inside[j]]:
                if not native.owners[int(owner)]['closed']:continue
                faces=native.owners[int(owner)]['ids']
                closest,d2=native._finite_witness(np.broadcast_to(point,(len(faces),3)),faces)
                finite_distance_evaluations+=len(faces)
                win=np.lexsort((faces,d2))[0];distance=float(np.sqrt(d2[win]))
                pad=float(2048*np.finfo(float).eps*(np.linalg.norm(point)+scale+cap+1))
                if np.isfinite(distance) and distance>cap+pad:
                    witnesses.append(dict(site=int(site_ids[j]),alpha=float(alphas[j]),point_mm=point.tolist(),
                        owner=int(owner),closed_owner=True,original_containment_valid=True,
                        independent_oblique_containment_valid=True,independent_owner_membership_matches=True,
                        finite_penetration_depth_mm=distance,penetration_cap_mm=cap,rounding_pad_mm=pad,
                        closest_filtered_face=int(faces[win]),triangle_vertex_ids=native.faces[faces[win]].tolist(),
                        closest_point_mm=closest[win].tolist(),all_owner_facets_enumerated=len(faces)))
                    found=True
        if not found:unresolved.append(dict(site=int(site_ids[j]),alpha=float(alphas[j]),signs_verified=signs_ok))
    return dict(classification='genuine_cap_crossing' if witnesses else 'inconclusive',
        genuine_cap_crossing=bool(witnesses),witnesses=witnesses,checked_sites=count,
        unclassified_checked_sites=unresolved,finite_distance_evaluations=finite_distance_evaluations,
        scope='At least one actual closed-owner finite cap crossing; not a statement that every invalid query was explained')


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
    started=time.perf_counter();raw=a.mechanics_report.read_bytes();report=json.loads(raw);run=report['runs'][a.case]
    prior_raw=a.previous_audit.read_bytes();prior=json.loads(prior_raw)
    if prior.get('qualified') is not True or prior.get('case')!=a.case:raise ValueError('Matching qualified prior interval audit required')
    if sha(prior['accepted_npz_snapshot_path'])!=prior['accepted_npz_sha256'] or sha(prior['mechanics_report_path'])!=prior['mechanics_report_sha256']:
        raise ValueError('Prior accepted audit snapshots changed')
    with np.load(prior['accepted_npz_snapshot_path']) as z:
        q0=z['positions_mm'][-1];v0=z['velocities_mm_s'][-1];t0=float(z['physical_times_s'][-1])
    rows=run['reports'];candidate_meta=None;metadata_raw=None;failed=None
    if a.failed_audit:
        failed_raw=a.failed_audit.read_bytes();failed=json.loads(failed_raw)
        if failed.get('qualified') is not False or failed.get('case')!=a.case:raise ValueError('Matching failed external audit required')
        if not failed.get('contact_sources_unchanged_during_audit') or not failed.get('audited_snapshots_unchanged_during_audit'):
            raise ValueError('Failed external audit has an integrity problem, not a geometric classification target')
        if sha(failed['mechanics_report_path'])!=failed['mechanics_report_sha256']:
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
    report_copy.write_bytes(raw);state_copy.write_bytes(data)
    if metadata_raw is not None:
        metadata_copy=out/(a.output_tag+'-checkpoint-metadata.json');metadata_copy.write_bytes(metadata_raw)
    solid=load_geometry_with_source_fibres(HERE.parent/'geometry/whole-face-continuum-final.npz',HERE.parent/'geometry/whole-face-muscle-phases-final.npz')
    supports,_=load_support_bundle(solid,HERE.parent/'mechanics/coarse-physical-site-v1/supports')
    guard=create_guard(solid,supports);native=guard.rigid.native
    expected=report['coarse_contact_provenance']
    for key,value in guard.provenance.items():
        if key.endswith('_sha256') and expected.get(key)!=value:raise ValueError('Changed contact/query source: '+key)
    for key,path in [('source_domain_sha256',HERE.parent/'geometry/whole-face-continuum-final.npz'),
                     ('source_phase_sha256',HERE.parent/'geometry/whole-face-muscle-phases-final.npz'),
                     ('source_support_sha256',HERE.parent/'mechanics/coarse-physical-site-v1/supports/clipped-origin-support.json')]:
        if report[key]!=prior[key]:raise ValueError('Changed prior physical source: '+key)
        if sha(path)!=report[key]:raise ValueError('Actual physical source differs from report: '+key)
        if failed is not None and failed[key]!=report[key]:raise ValueError('Failed audit physical source mismatch: '+key)
    for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
        if expected[key]!=prior['physical_query_data_sha256'][key]:raise ValueError('Changed prior query data: '+key)
    if failed is None:
        samples=row['frame_path'].get('samples')
        if not isinstance(samples,list) or not samples:raise ValueError('No failed native path sample to classify')
        count=max(1,int(np.ceil(guard.max_physical_site_difference(q0,q1)/guard.path_spacing)))
        alpha=(len(samples)-1)/count;state=guard._state(q0+alpha*(q1-q0))
        ids=np.unique(np.r_[state['ids'][~state['valid']],state['force_ids'][state['force_gap']<=-guard.penetration_limit]])
        points=state['points'][ids];alphas=np.full(len(ids),alpha)
    else:
        offenders=interval['site_chord'].get('offenders',[])
        ids=np.array([x['site'] for x in offenders],int);alphas=np.array([x['alpha'] for x in offenders],float)
        P0,P1=guard.C[ids]@q0,guard.C[ids]@q1;points=P0+alphas[:,None]*(P1-P0)
        if any(not np.allclose(point,offender['point_mm'],rtol=0,atol=1e-11) for point,offender in zip(points,offenders)):
            raise ValueError('External offender coordinate differs from its actual affine chord')
    evidence=finite_cap_witnesses(native,points,ids,alphas,guard.penetration_limit)
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
        candidate_npz_snapshot_path=str(state_copy),candidate_npz_sha256=sha(state_copy),
        mechanics_report_snapshot_path=str(report_copy),mechanics_report_sha256=sha(report_copy),
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
    evidence['snapshots_unchanged']=sha(state_copy)==hashlib.sha256(data).hexdigest() and sha(report_copy)==hashlib.sha256(raw).hexdigest()
    if not evidence['snapshots_unchanged']:raise ValueError('Classification snapshots changed')
    result_path.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(classification_path=str(result_path),classification_sha256=sha(result_path),classification=evidence['classification'],witnesses=len(evidence['witnesses']),wall_seconds=evidence['wall_seconds'])),flush=True)
    if not evidence['genuine_cap_crossing']:raise SystemExit(2)


if __name__=='__main__':main()
