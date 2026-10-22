"""Inspect one newest exact matched pair without repeatedly encoding its prefix.

Only full validated immutable histories enter. The two outputs are explicitly
single-state inspections, not an animation. No physical sample is synthesized.
"""
import argparse
import hashlib
import json
from pathlib import Path
from renderer_report_binding import recheck_report_bindings
import time
import numpy as np
from literal_renderer import Scene,load_geometry,sha_file,sha_array,reconstruct_boundary
from accepted_frames import load_accepted_frames,boundary_frame
from compare_actual import match_times,motion_metrics,comparable_contact_provenance
from render_mobile_actual import mobile_camera,render_mobile
from paired_cursor_binding import read_paired_cursor
from fixed_policy_reader import recheck_fixed_plan_bindings


def cursor_pair_inputs(cursor):
    t=cursor.get('accepted_physical_time_s',float('nan'))
    if not np.isfinite(t) or t<=0:raise ValueError('Cursor has no positive admitted physical time')
    audits=cursor.get('accepted_interval_audits',{})
    paths=[]
    for case in ('zygomaticus_frontalis','passive'):
        path=audits.get(case)
        if not isinstance(path,str) or not path:raise ValueError('Cursor lacks the admitted paired audit paths')
        paths.append(Path(path))
    return paths,float(t)


def numerical_pair_contrast(active_audit,passive_audit,active_index,passive_index):
    arows=active_audit['accepted_frame_evidence'];prows=passive_audit['accepted_frame_evidence']
    ar,pr=arows[active_index],prows[passive_index]
    amu=ar['contact_accounting']['barrier_scale'];pmu=pr['contact_accounting']['barrier_scale']
    fixed_pair=ar.get('barrier_policy') is not None or pr.get('barrier_policy') is not None
    values={'passive_final_mu':pmu,'active_final_mu':amu,
        'passive_lower_mu_sensitivity_mm':pr['lower_mu_site_difference_mm'],
        'active_lower_mu_sensitivity_mm':ar['lower_mu_site_difference_mm']}
    declared=active_audit.get('paired_numerical_parameter_contrast') or passive_audit.get('paired_numerical_parameter_contrast')
    return {**values,'passive_barrier_policy':pr.get('barrier_policy','legacy_continuation'),
        'active_barrier_policy':ar.get('barrier_policy','legacy_continuation'),
        'passive_fresh_lower_mu_comparison':pr.get('lower_mu_comparison_passed'),
        'active_fresh_lower_mu_comparison':ar.get('lower_mu_comparison_passed'),
        'same_final_barrier_scale':amu==pmu,
        'cumulative_barrier_scale_histories_equal':[r['contact_accounting']['barrier_scale'] for r in arows[:active_index+1]]==
            [r['contact_accounting']['barrier_scale'] for r in prows[:passive_index+1]],
        'both_cases_use_the_same_fixed_continuation_schedule':None if fixed_pair or active_audit.get('barrier_scales') is None or passive_audit.get('barrier_scales') is None else active_audit['barrier_scales']==passive_audit['barrier_scales'],
        'both_cases_use_the_same_fixed_policy':ar.get('barrier_policy')==pr.get('barrier_policy') if fixed_pair else None,
        'cumulative_physical_sample_times_equal':np.array_equal(active_audit['physical_times_s'][:active_index+1],passive_audit['physical_times_s'][:passive_index+1]),
        'declared_paired_numerical_parameter_contrast':declared,
        'declared_values_match_current_pair':None if declared is None else all(declared.get(k)==v for k,v in values.items()),
        'exact_isolated_muscle_effect_claim':False,
        'scope':'Literal active-minus-passive case contrast includes any disclosed numerical-parameter/history differences; it is not an exact isolated muscle effect.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--active-audit',type=Path)
    parser.add_argument('--passive-audit',type=Path)
    parser.add_argument('--paired-cursor',type=Path,help='Read the admitted paired cursor once and retain that exact byte snapshot')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output directory to preserve evidence')
    cursor_bytes=None;cursor_time=None;cursor_snapshot=None
    if args.paired_cursor:
        if args.active_audit or args.passive_audit:parser.error('Use either a paired cursor or two explicit receipts')
        cursor_snapshot=read_paired_cursor(args.paired_cursor);cursor_bytes=cursor_snapshot['raw_bytes']
        (args.active_audit,args.passive_audit),cursor_time=cursor_pair_inputs(json.loads(cursor_bytes))
    elif not (args.active_audit and args.passive_audit):parser.error('Supply a paired cursor or both explicit receipts')
    started=time.perf_counter();g=load_geometry(args.geometry);loaded=[];immutable=[(args.geometry,g['sha256'])]
    for receipt,case in ((args.active_audit,'zygomaticus_frontalis'),(args.passive_audit,'passive')):
        proof=json.loads(receipt.read_text())
        frame_path=Path(proof['accepted_npz_snapshot_path']);report_path=Path(proof['mechanics_report_path'])
        frames,audit=load_accepted_frames(frame_path,report_path,case,g,receipt,cursor_snapshot);loaded.append((frames,audit))
        immutable.extend(((receipt,audit['external_closed_site_path_audit']['audit_sha256']),
            (frame_path,audit['frame_sha256']),(report_path,audit['report_sha256'])))
    (a,aa),(p,pa)=loaded
    if not np.array_equal(a['source_initial_positions_mm'],p['source_initial_positions_mm']) or \
            float(a['scale_mm_per_L0'])!=float(p['scale_mm_per_L0']):
        raise ValueError('Matched comparison requires identical source coordinates and physical units')
    for key in ('source_phase_sha256','contact_mode','contact_enforcement','site_resolution_mm',
                'exterior_contact_stationarity','epigraph_mechanics_source_sha256'):
        if aa.get(key)!=pa.get(key):raise ValueError(f'Matched cases differ in {key}')
    if comparable_contact_provenance(aa['coarse_contact_provenance'])!=comparable_contact_provenance(pa['coarse_contact_provenance']):
        raise ValueError('Matched cases differ in physical contact provenance')
    times,ia,ip=match_times(a['physical_times_s'],p['physical_times_s']);t=float(times[-1]);ai,pi=int(ia[-1]),int(ip[-1])
    if cursor_time is not None and (t!=cursor_time or a['physical_times_s'][-1]!=t or p['physical_times_s'][-1]!=t):
        raise ValueError('Frozen receipts do not exactly match the cursor admitted paired endpoint')
    camera=mobile_camera(g['vertices']*float(a['scale_mm_per_L0']))
    scene=Scene(g,scale_mm_per_L0=float(a['scale_mm_per_L0']),cameras=[camera])
    av,pv=boundary_frame(a,ai,g),boundary_frame(p,pi,g)
    camera.check_boundary(av);camera.check_boundary(pv);args.output.mkdir(parents=True)
    images=[];outputs=[]
    for label,case,vertices,frames,index in (('active','zygomaticus_frontalis',av,a,ai),('passive','passive',pv,p,pi)):
        ts=time.perf_counter();im,stats=render_mobile(scene,vertices,t,case);path=args.output/(label+'.png');im.save(path);images.append(im)
        outputs.append({'image':path.name,'sha256':sha_file(path),'case':case,'source_frame_index':index,'physical_time_s':t,
            'q_sha256':sha_array(frames['positions_mm'][index]),'boundary_xyz_sha256':sha_array(vertices),'render':stats,
            'render_and_png_seconds':time.perf_counter()-ts})
    metrics=motion_metrics(scene,av,pv,*images,front_panel_box=(40,148,1040,1308))
    reference=reconstruct_boundary(a['source_initial_positions_mm'],g['tets'],g['exterior_tet'],g['exterior_barycentric'])
    reference_image,reference_stats=render_mobile(scene,reference,0.,'source_initial',source_initial_reference=True)
    reference_path=args.output/'source-initial-reference.png';reference_image.save(reference_path)
    rest_motion={}
    for label,vertices,current_image in (('active',av,images[0]),('passive',pv,images[1])):
        measured=motion_metrics(scene,vertices,reference,current_image,reference_image,front_panel_box=(40,148,1040,1308))
        measured['current_minus_source_initial_exterior_vertex_displacement_mm']=measured.pop('active_minus_passive_exterior_vertex_displacement_mm')
        measured['front_visible_exterior_faces']={'current':measured['front_visible_exterior_faces']['active'],
            'source_initial':measured['front_visible_exterior_faces']['passive']}
        rest_motion[label]=measured
    rest_motion['source_initial_is_not_equilibrated_neutral']=True
    rest_motion['passive_motion_includes_gravity_settling']=True
    rest_motion['active_motion_is_not_isolated_muscle_response']=True
    parameter_contrast=numerical_pair_contrast(aa,pa,ai,pi)
    for path,expected in immutable:
        if sha_file(path)!=expected:raise RuntimeError('Immutable input changed; inspection is not qualified')
    recheck_fixed_plan_bindings(aa);recheck_fixed_plan_bindings(pa)
    recheck_report_bindings(aa);recheck_report_bindings(pa)
    report={'active_input':aa,'passive_input':pa,'scene':scene.metadata(),'outputs':outputs,
        'physical_time_s':t,'exact_matched_source_indices':{'active':ai,'passive':pi},'metrics':metrics,
        'paired_numerical_parameter_contrast':parameter_contrast,
        'literal_source_initial_to_current_motion':rest_motion,
        'source_initial_reference':{'file':reference_path.name,'sha256':sha_file(reference_path),
            'q_sha256':sha_array(a['source_initial_positions_mm']),'boundary_xyz_sha256':sha_array(reference),
            'actual_physical_time_s':0.,'not_an_accepted_forward_step':True,'render':reference_stats},
        'single_pair_inspection_is_not_an_animation':True,'history_was_not_resampled':True,
        'inspection_source_sha256':sha_file(Path(__file__)),
        'processing_seconds':time.perf_counter()-started}
    if cursor_bytes is not None:
        snapshot=args.output/'paired-cursor-snapshot.json';snapshot.write_bytes(cursor_bytes)
        report['paired_cursor']={'original_path':str(args.paired_cursor.resolve()),'snapshot':snapshot.name,
            'sha256':hashlib.sha256(cursor_bytes).hexdigest(),'accepted_physical_time_s':cursor_time,
            'live_cursor_need_not_remain_unchanged_after_snapshot':True}
    (args.output/'comparison-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output.resolve()),'physical_time_s':t,'metrics':metrics,
        'paired_numerical_parameter_contrast':parameter_contrast,
        'literal_source_initial_to_current_motion':rest_motion},indent=2))


if __name__=='__main__':main()
