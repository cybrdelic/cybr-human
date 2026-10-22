"""Hash-bound external audit of genuinely saved physical results, no dynamics."""
from pathlib import Path
import argparse,hashlib,io,json,time
import numpy as np
from coarse_physical_site_contact_epigraph import create_guard
from coarse_physical_site_contact import sha
from continuous_closed_site_path import certify_closed_owner_site_chord
from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle
from volume_guard import certify_positive_volume

HERE=Path(__file__).resolve().parent


def array_sha(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mechanics-report',type=Path,required=True)
    p.add_argument('--case',choices=['passive','zygomaticus_frontalis'],required=True)
    p.add_argument('--output-tag',required=True)
    a=p.parse_args()
    if not a.output_tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.output_tag):
        p.error('A simple new output tag is required')
    folder=HERE/'temporal-audits';folder.mkdir(exist_ok=True)
    output=folder/(a.output_tag+'.json')
    report_copy=folder/(a.output_tag+'-mechanics-report.json')
    state_copy=folder/(a.output_tag+'-accepted.npz')
    if any(x.exists() for x in (output,report_copy,state_copy)):p.error('New output tag required; preserve earlier audit')
    started=time.perf_counter()
    report_bytes=a.mechanics_report.read_bytes();report=json.loads(report_bytes)
    if report.get('initial_velocity')!='zero':
        raise ValueError('This source-history schema requires the declared zero initial velocity')
    run=report['runs'][a.case]
    if run.get('accepted_frames',0)<1:
        raise ValueError('No genuinely accepted physical frame exists in this case')
    source=Path(run['accepted_npz_path'])
    data=source.read_bytes();data_sha=hashlib.sha256(data).hexdigest()
    if data_sha!=run['accepted_npz_sha256'] or run['accepted_frames']<1:
        raise ValueError('A hash-matching genuinely accepted physical result is required')
    with np.load(io.BytesIO(data)) as z:
        q=z['positions_mm'];v=z['velocities_mm_s'];times=z['physical_times_s']
        initial=z['source_initial_positions_mm'];scale=float(z['scale_mm_per_L0'])
    if (len(q)!=run['accepted_frames'] or q.shape!=v.shape or q.ndim!=3 or q.shape[2]!=3
            or times.shape!=(len(q),) or not np.isfinite(q).all() or not np.isfinite(v).all()
            or not np.isfinite(times).all() or np.any(np.diff(np.r_[0.,times])<=0)):
        raise ValueError('Malformed accepted physical history')
    accepted_rows=[row for row in run['reports'] if row['accepted']]
    if len(accepted_rows)!=len(q) or run['actual_times_s']!=times.tolist():
        raise ValueError('Report and saved accepted history differ')
    for i,row in enumerate(accepted_rows):
        dt=float(times[i]-(0. if i==0 else times[i-1]))
        previous=initial if i==0 else q[i-1]
        if (not np.isclose(dt,report['physical_dt_s'],rtol=1e-12,atol=1e-15)
                or not np.isclose(dt,row['dt_s'],rtol=1e-12,atol=1e-15)
                or row['physical_time_s']!=float(times[i])
                or not np.allclose(v[i],(q[i]-previous)/dt,rtol=2e-12,atol=2e-12)):
            raise ValueError('Saved velocity or report time differs from the physical chord history')
    report_copy.write_bytes(report_bytes);state_copy.write_bytes(data)
    solid=load_geometry_with_source_fibres(HERE.parent/'geometry/whole-face-continuum-final.npz',HERE.parent/'geometry/whole-face-muscle-phases-final.npz')
    if not np.array_equal(initial,solid.X) or scale!=solid.mm_per_L0:
        raise ValueError('Accepted source/scale differs from actual carrier')
    for field,file in [('source_domain_sha256',HERE.parent/'geometry/whole-face-continuum-final.npz'),
                       ('source_phase_sha256',HERE.parent/'geometry/whole-face-muscle-phases-final.npz'),
                       ('source_support_sha256',HERE.parent/'mechanics/coarse-physical-site-v1/supports/clipped-origin-support.json')]:
        if report[field]!=sha(file):raise ValueError('Changed physical source: '+field)
    expected=report['coarse_contact_provenance']
    sources={}
    for field,file in [('exterior_branch_provider_sha256','coarse_physical_site_contact_epigraph.py'),
                       ('exterior_branch_source_sha256','exterior_feature_bundles.py'),
                       ('base_contact_source_sha256','coarse_physical_site_contact.py'),
                       ('rigid_query_source_sha256','coarse_rigid_query.py'),
                       ('compact_barrier_source_sha256','compact_penetration_barrier.py'),
                       ('rolling_adapter_source_sha256','coarse_physical_site_contact_rolling.py'),
                       ('rolling_screen_source_sha256','rolling_exterior_bounds.py')]:
        sources[file]=sha(HERE/file)
        if expected[field]!=sources[file]:raise ValueError('Changed contact source: '+file)
    sources['continuous_closed_site_path.py']=sha(HERE/'continuous_closed_site_path.py')
    sources['audit_accepted_site_chords.py']=sha(Path(__file__))
    sources['volume_guard.py']=sha(HERE/'volume_guard.py')
    supports,_=load_support_bundle(solid,HERE.parent/'mechanics/coarse-physical-site-v1/supports')
    guard=create_guard(solid,supports)
    for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256'):
        if guard.provenance[key]!=expected[key]:raise ValueError('Changed physical query data: '+key)
    result=dict(schema='continuous_closed_site_external_audit_v1',qualified=False,case=a.case,
        accepted_npz_path=str(source.resolve()),accepted_npz_sha256=data_sha,
        accepted_npz_snapshot_path=str(state_copy.resolve()),
        mechanics_report_path=str(report_copy.resolve()),mechanics_report_sha256=sha(report_copy),
        mechanics_report_sha256_at_audit=sha(report_copy),
        original_mechanics_report_path=str(a.mechanics_report.resolve()),
        accepted_frames_audited=0,physical_times_s=times.tolist(),
        source_initial_positions_sha256=array_sha(initial),contact_sources_sha256=sources,
        array_hash_convention='SHA256(str(array.dtype).encode() + str(array.shape).encode() + C-contiguous array.tobytes())',
        source_domain_sha256=report['source_domain_sha256'],source_phase_sha256=report['source_phase_sha256'],
        source_support_sha256=report['source_support_sha256'],site_resolution_mm=scale,
        physical_query_data_sha256={key:guard.provenance[key] for key in ('source_cells_sha256','native_source_sha256','reference_band_cache_sha256')},
        saved_velocity_history_consistent=True,
        global_surface_collision_certified=False,unsigned_open_sheet_non_crossing_certified=False,
        scope='External continuous CLOSED-owner penetration-cap check of saved occupied-site chords, plus existing positive-J path certificate',
        physical_force_law_changed=False,physical_history_changed=False,intervals=[])
    for i in range(len(q)):
        previous=initial if i==0 else q[i-1]
        previous_velocity=np.zeros_like(initial) if i==0 else v[i-1]
        closed=certify_closed_owner_site_chord(guard,previous,q[i])
        volume=certify_positive_volume(solid,previous,q[i],floor=guard.jacobian_floor)
        row=dict(frame_index=i,t0_s=0. if i==0 else float(times[i-1]),t1_s=float(times[i]),
            q0_sha256=array_sha(previous),q1_sha256=array_sha(q[i]),
            previous_velocity_sha256=array_sha(previous_velocity),site_chord=closed,positive_J_path=volume,
            qualified=bool(closed['continuous_closed_owner_site_cap_checked'] and volume['certified']))
        result['intervals'].append(row);result['accepted_frames_audited']=i+1
        print(json.dumps(row),flush=True)
        if not row['qualified']:break
    result['original_source_unchanged_during_audit']=sha(source)==data_sha
    result['audited_snapshots_unchanged_during_audit']=(sha(state_copy)==data_sha
        and sha(report_copy)==result['mechanics_report_sha256'])
    result['contact_sources_unchanged_during_audit']=all(sha(HERE/file)==value for file,value in sources.items())
    result['qualified']=bool(len(result['intervals'])==len(q) and all(r['qualified'] for r in result['intervals'])
        and result['audited_snapshots_unchanged_during_audit'] and result['contact_sources_unchanged_during_audit'])
    result['wall_seconds']=time.perf_counter()-started
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(audit_path=str(output),audit_sha256=sha(output),qualified=result['qualified'],wall_seconds=result['wall_seconds'])),flush=True)
    if not result['qualified']:raise SystemExit(2)


if __name__=='__main__':main()
