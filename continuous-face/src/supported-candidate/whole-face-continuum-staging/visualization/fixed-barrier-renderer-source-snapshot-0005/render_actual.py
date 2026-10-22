"""Render only real saved, accepted mechanics frames. No frames exist by default.

Optional movie uses exactly one encoded image per accepted frame, no looping,
interpolation, duplicate stills or apparent-motion gain. Actual physical time
and playback rate are shown. This command was NOT run on synthetic motion.
"""
import argparse
import json
from pathlib import Path
from renderer_report_binding import recheck_report_bindings
import subprocess
import time
from fractions import Fraction
import numpy as np
from literal_renderer import load_geometry, Scene, sha_file, sha_array
from accepted_frames import load_accepted_frames, boundary_frame

def movie_timing(times,positions,fps):
    if len(times)<2: raise ValueError('A single accepted still is not an animation')
    if any(np.array_equal(positions[i-1],positions[i]) for i in range(1,len(times))):
        raise ValueError('Movie export refuses consecutive identical coordinate frames')
    dt=np.diff(times)
    if not np.allclose(dt,dt[0],rtol=1e-9,atol=1e-12):
        raise ValueError('This movie interface requires uniform actual physical sampling; keep PNGs otherwise')
    if not np.isfinite(fps) or fps<=0: raise ValueError('Positive explicit playback fps required')
    return float(dt[0]*fps)

def verify_encoded_movie(path, expected_frames, expected_fps, expected_size):
    """Read the encoded video, rather than assuming an ffmpeg exit is enough."""
    command=['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=codec_name,width,height,nb_read_frames,r_frame_rate,avg_frame_rate,duration',
        '-of','json',str(path)]
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    streams=json.loads(result.stdout).get('streams',[])
    if len(streams)!=1:
        raise RuntimeError('Expected exactly one encoded video stream')
    stream=streams[0]
    if int(stream.get('nb_read_frames',-1))!=expected_frames:
        raise RuntimeError('Encoded movie frame count differs from actual accepted states')
    if (stream.get('width'),stream.get('height'))!=tuple(expected_size):
        raise RuntimeError('Encoded movie dimensions differ from the literal rendered images')
    fps=float(Fraction(stream.get('avg_frame_rate','0')))
    if not np.isclose(fps,expected_fps,rtol=1e-6,atol=1e-9):
        raise RuntimeError('Encoded movie playback rate differs from its disclosure')
    return {'ffprobe_command':command,'stream':stream,'decoded_frame_count_verified':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',required=True,type=Path)
    p.add_argument('--frames',required=True,type=Path)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--closed-path-audit',required=True,type=Path)
    p.add_argument('--case',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--movie',action='store_true')
    p.add_argument('--playback-fps',type=float,default=24.)
    args=p.parse_args()
    if args.output.exists(): p.error('Use a new output directory to preserve evidence')
    g=load_geometry(args.geometry)
    frames,audit=load_accepted_frames(args.frames,args.report,args.case,g,args.closed_path_audit)
    ts=frames['physical_times_s']; note=''; speed=None
    if args.movie:
        speed=movie_timing(ts,frames['positions_mm'],args.playback_fps)
        note=f'Playback {args.playback_fps:g} fps; {speed:.9g} physical seconds per screen second; no interpolated frames'
    scene=Scene(g,scale_mm_per_L0=float(frames['scale_mm_per_L0']))
    # Validate entire sequence coverage before exporting any image. No view
    # fitting to q; if needed, change the one fixed camera explicitly for all.
    for i in range(len(ts)):
        vertices=boundary_frame(frames,i,g)
        for c in scene.cameras: c.check_boundary(vertices)
    args.output.mkdir(parents=True)
    audit.update(scene=scene.metadata(),outputs=[],movie=None)
    for i,t in enumerate(ts):
        start=time.perf_counter(); vertices=boundary_frame(frames,i,g)
        image,stats=scene.render(vertices,reference_only=False,physical_time_s=float(t),case=args.case,playback_note=note,
            contact_mode=audit['contact_mode'],site_resolution_mm=audit['site_resolution_mm'],
            contact_enforcement=audit['contact_enforcement'])
        path=args.output/f'frame-{i:06d}.png'; image.save(path)
        audit['outputs'].append({'image':path.name,'image_sha256':sha_file(path),
            'q_sha256':sha_array(frames['positions_mm'][i]),'boundary_xyz_sha256':sha_array(vertices),
            'physical_time_s':float(t),'total_seconds':time.perf_counter()-start,'render':stats})
        print(f'Accepted frame {i+1}/{len(ts)}: actual t={t:.9g}s',flush=True)
    if args.movie:
        movie=args.output/'accepted-physical-frames.mp4'
        command=['/usr/bin/ffmpeg','-nostdin','-v','error','-framerate',str(args.playback_fps),
            '-i',str(args.output/'frame-%06d.png'),'-frames:v',str(len(ts)),
            '-fps_mode','passthrough','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
        subprocess.run(command,check=True)
        audit['movie']={'file':movie.name,'sha256':sha_file(movie),'command':command,
            'encoded_frames':len(ts),'playback_fps':args.playback_fps,
            'physical_seconds_per_screen_second':speed,'frame_interpolation':False,'looping':False,
            'physical_first_and_last_sample_s':[float(ts[0]),float(ts[-1])],
            'encoding_verification':verify_encoded_movie(movie,len(ts),args.playback_fps,(2250,1200))}
    # Verify the exact byte inputs did not change during rendering.
    for path,expected in [(args.geometry,g['sha256']),(args.frames,audit['frame_sha256']),(args.report,audit['report_sha256']),
                          (args.closed_path_audit,audit['external_closed_site_path_audit']['audit_sha256'])]:
        if sha_file(path)!=expected: raise RuntimeError('Input changed during rendering; outputs not qualified')
    recheck_report_bindings(audit)
    (args.output/'render-audit.json').write_text(json.dumps(audit,indent=2)+'\n')

if __name__=='__main__': main()
