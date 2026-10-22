"""Uncluttered portrait presentation of real, externally audited states only.

The source-fixed front camera enlarges the entire face uniformly for a mobile
viewport. It never amplifies displacement relative to the face, fits a frame,
changes coordinates, smooths geometry or invents intermediate physical times.
Detailed mechanics/contact disclosures are preserved in the companion JSON.
"""
import argparse
import json
from pathlib import Path
from renderer_report_binding import recheck_report_bindings
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw
from literal_renderer import Camera, Scene, load_geometry, project, font, sha_file, sha_array
from accepted_frames import load_accepted_frames, boundary_frame
from render_actual import movie_timing, verify_encoded_movie
from variable_movie import variable_movie_timing, encode_variable_movie
from paired_cursor_binding import read_paired_cursor
from fixed_policy_reader import recheck_fixed_plan_bindings


def mobile_camera(source_reference, width=1000, height=1160, padding=.08):
    projected=project(source_reference,0.)
    if width<1 or height<1 or not 0<padding<1:
        raise ValueError('Invalid fixed mobile viewport')
    lower,upper=projected[:,:2].min(0),projected[:,:2].max(0)
    span=(upper-lower)*(1+2*padding)
    scale=min(width/max(span[0],1e-12),height/max(span[1],1e-12))
    center=(lower+upper)/2
    low,high=np.percentile(projected[:,2],[2,98])
    return Camera(0.,width,height,float(center[0]),float(center[1]),float(scale),float(low),float(high))


def render_mobile(scene,vertices,time_s,case,playback_note='',*,source_initial_reference=False):
    if len(scene.cameras)!=1 or not np.isfinite(time_s) or (time_s<=0 and not (time_s==0 and source_initial_reference)) or (source_initial_reference and time_s!=0):
        raise ValueError('Require a positive saved physical time, or an explicitly labeled actual t=0 source reference')
    camera=scene.cameras[0]
    buffers=scene.raster.buffers(camera)
    boundary_stats=scene.raster.draw(buffers,camera.check_boundary(vertices),scene.g['triangles'],
        scene.g['boundary_class'],camera)
    bone_stats=scene.raster.draw(buffers,scene.bone_projections[0],scene.bone_faces,scene.bone_classes,camera,
        palette=scene.bone_palette,depth_cue=False,owner_offset=len(scene.g['triangles']))
    out=Image.new('RGB',(1080,1440),'white')
    out.paste(Image.fromarray(buffers[0]),(40,148))
    draw=ImageDraw.Draw(out)
    draw.text((540,35),'CYBR HUMAN',font=font(33),fill='#20313d',anchor='mt')
    label='Source initial state' if source_initial_reference else ('Passive' if case=='passive' else ('Zygomaticus + frontalis' if case=='zygomaticus_frontalis' else case))
    draw.text((540,88),f'{label}  |  actual t = {time_s:.6g} s',font=font(22),fill='#354551',anchor='mt')
    if playback_note:
        draw.text((540,121),playback_note,font=font(18),fill='#52616a',anchor='mt')
    draw.text((540,1340),'Coarse physical model · motion shown without amplification',
        font=font(21),fill='#52616a',anchor='mt')
    draw.text((540,1379),'Model and contact limitations are in the accompanying report.',
        font=font(18),fill='#6c787f',anchor='mt')
    return out,{'boundary':boundary_stats,'bones':bone_stats,
                'pixels_visible':int(np.isfinite(buffers[1]).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--closed-path-audit',type=Path,required=True)
    parser.add_argument('--paired-cursor',type=Path,help='Required for the separately versioned 0.02 mm reconstruction contract')
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--movie',action='store_true')
    parser.add_argument('--playback-fps',type=float,default=24.)
    parser.add_argument('--slow-motion-factor',type=float,
        help='Use proportional variable frame timestamps with this single explicit slowdown factor')
    args=parser.parse_args()
    processing_started=time.perf_counter()
    if args.output.exists():
        parser.error('Use a new output directory to preserve evidence')
    geometry=load_geometry(args.geometry)
    cursor=None if args.paired_cursor is None else read_paired_cursor(args.paired_cursor)
    frames,audit=load_accepted_frames(args.frames,args.report,args.case,geometry,args.closed_path_audit,cursor)
    times=frames['physical_times_s']
    speed=None;note='';variable_timing=None
    if args.movie:
        if args.slow_motion_factor is not None:
            variable_timing=variable_movie_timing(times,frames['positions_mm'],args.slow_motion_factor)
            note=f'{args.slow_motion_factor:g}× slow motion · actual sample timing preserved'
        else:
            speed=movie_timing(times,frames['positions_mm'],args.playback_fps)
            note=f'{args.playback_fps:g} fps playback · {speed:.6g} physical seconds per screen second'
    scale=float(frames['scale_mm_per_L0'])
    camera=mobile_camera(geometry['vertices']*scale)
    scene=Scene(geometry,scale_mm_per_L0=scale,cameras=[camera])
    for i in range(len(times)):
        camera.check_boundary(boundary_frame(frames,i,geometry))
    args.output.mkdir(parents=True)
    if cursor is not None:
        (args.output/'paired-cursor-snapshot.json').write_bytes(cursor['raw_bytes'])
    audit.update(presentation='Fixed portrait front view; whole-face framing only, no displacement amplification',
        presentation_source_sha256=sha_file(Path(__file__)),
        scene=scene.metadata(),outputs=[],movie=None,
        limitations={'coarse_physical_site_resolution_mm':audit['site_resolution_mm'],
            'global_surface_collision_certified':False,'unsigned_open_sheet_non_crossing_certified':False,
            'positive_mu_complementarity_is_approximate':True,
            'engineering_scale_is_not_patient_calibration':True,
            'source_registered_transient_not_equilibrated_neutral':True,
            'one_frame_is_not_an_animation':True,
            'small_early_motion_is_not_a_visible_expression_claim':True})
    audit['limitations']['interframe_geometry']={
        'reconstruction_contract':audit.get('reconstruction_contract'),
        'reconstruction_contract_schema':audit.get('reconstruction_contract_schema'),
        'trajectory_error_bound_proven':audit.get('trajectory_error_bound_proven'),
        'physical_endpoint_penetration_cap_mm':audit.get('physical_endpoint_penetration_cap_mm'),
        'fixed_interframe_geometric_allowance_mm':audit.get('interframe_reconstruction_allowance_mm'),
        'certified_interframe_penetration_cap_mm':audit.get('certified_interframe_penetration_cap_mm'),
        'not_a_total_temporal_or_state_error_bound':True}
    audit['limitations']['stationarity']={
        'policy':audit.get('stationarity_accuracy_policy'),
        'approximate_frame_indices':[i for i,r in enumerate(audit['validated_stationarity_routes']) if r['approximate_stationarity']],
        'absolute_force_target_misses_are_explicit_in_accepted_frame_evidence':True,
        'full_qp_force_state_fallback_is_a_local_indicator_not_a_total_state_or_trajectory_error_bound':True}
    audit['limitations']['barrier_sensitivity']=audit['barrier_sensitivity_scope']
    audit['limitations']['case_comparison']={
        'paired_numerical_parameter_contrast':audit.get('paired_numerical_parameter_contrast'),
        'exact_isolated_muscle_effect_claim':False}
    for i,t in enumerate(times):
        render_started=time.perf_counter()
        vertices=boundary_frame(frames,i,geometry)
        result,stats=render_mobile(scene,vertices,float(t),args.case,note)
        path=args.output/f'frame-{i:06d}.png';result.save(path)
        audit['outputs'].append({'image':path.name,'sha256':sha_file(path),'physical_time_s':float(t),
            'q_sha256':sha_array(frames['positions_mm'][i]),'boundary_xyz_sha256':sha_array(vertices),'render':stats,
            'boundary_reconstruction_render_and_png_seconds':time.perf_counter()-render_started})
        print(json.dumps({'image':str(path.resolve()),'physical_time_s':float(t),'pixels_per_mm':camera.pixels_per_unit}),flush=True)
    if args.movie and variable_timing is not None:
        audit['movie']=encode_variable_movie(
            [args.output/row['image'] for row in audit['outputs']],args.output/'actual-face-motion.mp4',variable_timing,(1080,1440))
    elif args.movie:
        movie=args.output/'actual-face-motion.mp4'
        command=['/usr/bin/ffmpeg','-nostdin','-v','error','-framerate',str(args.playback_fps),
            '-i',str(args.output/'frame-%06d.png'),'-frames:v',str(len(times)),'-fps_mode','passthrough',
            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
        subprocess.run(command,check=True)
        audit['movie']={'file':movie.name,'sha256':sha_file(movie),'command':command,'encoded_frames':len(times),
            'playback_fps':args.playback_fps,'physical_seconds_per_screen_second':speed,
            'physical_first_and_last_sample_s':[float(times[0]),float(times[-1])],
            'frame_interpolation':False,'looping':False,
            'encoding_verification':verify_encoded_movie(movie,len(times),args.playback_fps,(1080,1440))}
    for path,expected in ((args.geometry,geometry['sha256']),(args.frames,audit['frame_sha256']),
            (args.report,audit['report_sha256']),(args.closed_path_audit,audit['external_closed_site_path_audit']['audit_sha256'])):
        if sha_file(path)!=expected:
            raise RuntimeError('Immutable rendering input changed; output is not qualified')
    if audit.get('admitted_paired_cursor') is not None:
        for proof in audit['admitted_paired_cursor']['selected_independent_proofs'].values():
            if sha_file(Path(proof['path']))!=proof['sha256']:
                raise RuntimeError('An immutable admitted paired proof changed during rendering')
    recheck_fixed_plan_bindings(audit)
    recheck_report_bindings(audit)
    audit['processing_seconds_excluding_final_audit_write']=time.perf_counter()-processing_started
    (args.output/'render-audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    main()
