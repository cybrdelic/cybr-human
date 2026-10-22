"""One actual image per saved state, with proportional variable timestamps.

Changing presentation timestamps does not interpolate positions or images.
The final encoded sample has one ordinary display duration, explicitly a
terminal hold, not an extra physical state or an extrapolated physical time.
"""
from platform_runtime import resolve_tool
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
from literal_renderer import sha_file

TIMESCALE=1_000_000


def variable_movie_timing(times,positions,slow_motion_factor):
    times=np.asarray(times,dtype=float)
    positions=np.asarray(positions)
    if times.ndim!=1 or len(times)<2 or positions.shape[0]!=len(times):
        raise ValueError('At least two distinct actual saved states are required')
    if not np.isfinite(times).all() or times[0]<=0 or np.any(np.diff(times)<=0):
        raise ValueError('Actual physical sample times must be finite, positive and strictly increasing')
    if not np.isfinite(slow_motion_factor) or slow_motion_factor<1:
        raise ValueError('Slow-motion factor must be finite and at least one')
    if not np.isfinite(positions).all() or any(np.array_equal(positions[i-1],positions[i]) for i in range(1,len(times))):
        raise ValueError('Movie refuses nonfinite or consecutive identical saved coordinates')
    exact=(times-times[0])*slow_motion_factor
    if not np.isfinite(exact).all() or exact[-1]*TIMESCALE>2**52:
        raise ValueError('Presentation timeline is not representable')
    ticks=np.rint(exact*TIMESCALE).astype(np.int64)
    differences=np.diff(ticks)
    if np.any(differences<=0):
        raise ValueError('Accepted intervals collapse at the encoded timebase; choose a larger single slow-motion factor')
    return {'slow_motion_factor':float(slow_motion_factor),
        'physical_seconds_per_screen_second':1./slow_motion_factor,
        'actual_physical_times_s':times.tolist(),'presentation_timestamps_ticks':ticks.tolist(),
        'presentation_timestamps_s':(ticks/TIMESCALE).tolist(),'video_timescale_ticks_per_second':TIMESCALE,
        'maximum_timestamp_quantization_error_s':float(np.max(np.abs(ticks/TIMESCALE-exact))),
        'sample_duration_ticks':np.r_[differences,differences[-1]].tolist(),
        'terminal_frame_hold_s':float(differences[-1]/TIMESCALE),
        'terminal_hold_is_new_physical_interval':False,
        'terminal_hold_disclosure':'The last encoded sample is displayed once for the preceding display interval. Its physical clock is fixed; no new physical state is implied.',
        'frame_interpolation':False,'looping':False,'duplicate_encoded_frames_added':False,
        'physical_first_and_last_sample_s':[float(times[0]),float(times[-1])]}


def verify_variable_movie(path,timing,expected_size):
    command=[resolve_tool('ffprobe'),'-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=pts,pts_time,duration_time',
        '-show_entries','stream=codec_name,profile,level,width,height,time_base,nb_frames,duration,avg_frame_rate,r_frame_rate',
        '-of','json',str(path)]
    result=json.loads(subprocess.run(command,check=True,capture_output=True,text=True).stdout)
    streams=result.get('streams',[]);frames=result.get('frames',[])
    if len(streams)!=1 or len(frames)!=len(timing['presentation_timestamps_ticks']):
        raise RuntimeError('Decoded variable-rate movie does not contain exactly one frame per accepted state')
    stream=streams[0]
    if (stream.get('width'),stream.get('height'))!=tuple(expected_size) or stream.get('time_base')!=f'1/{TIMESCALE}':
        raise RuntimeError('Encoded movie changed expected dimensions or precision timebase')
    pts=[frame.get('pts') for frame in frames]
    durations=[int(round(float(frame.get('duration_time','nan'))*TIMESCALE)) for frame in frames]
    if pts!=timing['presentation_timestamps_ticks'] or durations!=timing['sample_duration_ticks']:
        raise RuntimeError('Encoded timestamps or sample durations differ from the actual physical timing map')
    return {'ffprobe_command':command,'stream':stream,'decoded_frame_count_verified':True,
        'every_presentation_timestamp_verified':True,'every_sample_duration_verified':True,
        'decoded_presentation_timestamps_ticks':pts,'decoded_sample_duration_ticks':durations}


def encode_variable_movie(image_paths,movie,timing,expected_size):
    movie=Path(movie);image_paths=[Path(p) for p in image_paths]
    if movie.exists() or len(image_paths)!=len(timing['presentation_timestamps_ticks']):
        raise ValueError('New movie path and exactly one source image per saved state are required')
    manifest=movie.with_suffix('.ffconcat')
    if manifest.exists():
        raise ValueError('Preserve existing movie timing manifests')
    lines=['ffconcat version 1.0']
    source_images=[]
    for image,duration in zip(image_paths,timing['sample_duration_ticks']):
        if image.parent.resolve()!=movie.parent.resolve() or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in image.name):
            raise ValueError('Movie manifest requires safe sibling source-image names')
        if not image.is_file():
            raise ValueError('Every actual source image must exist')
        source_images.append({'file':image.name,'sha256':sha_file(image)})
        lines.extend([f"file '{image.name}'",f'option framerate {TIMESCALE}',f'duration {duration/TIMESCALE:.6f}'])
    manifest.write_text('\n'.join(lines)+'\n')
    # A correct finite nominal-rate hint avoids advertising a million-fps H264
    # level simply because the container and image demuxer use microsecond ticks.
    nominal_rate=max(1,int(math.ceil(TIMESCALE/min(timing['sample_duration_ticks']))))
    terminal_duration=timing['sample_duration_ticks'][-1]
    command=[resolve_tool('ffmpeg'),'-nostdin','-v','error','-safe','0','-i',str(manifest),
        '-frames:v',str(len(image_paths)),'-fps_mode','passthrough','-enc_time_base',f'1/{TIMESCALE}',
        '-c:v','libx264','-x264-params',f'fps={nominal_rate}/1:force-cfr=0','-bf','0','-crf','18',
        '-pix_fmt','yuv420p','-bsf:v',f"setts=duration='if(eq(N,{len(image_paths)-1}),{terminal_duration},DURATION)'",
        '-video_track_timescale',str(TIMESCALE),'-movflags','+faststart',str(movie)]
    started=time.perf_counter()
    subprocess.run(command,check=True)
    encode_seconds=time.perf_counter()-started
    started=time.perf_counter()
    verification=verify_variable_movie(movie,timing,expected_size)
    decode_verification_seconds=time.perf_counter()-started
    for image,row in zip(image_paths,source_images):
        if sha_file(image)!=row['sha256']:
            raise RuntimeError('A source image changed during encoding')
    return {**timing,'file':movie.name,'sha256':sha_file(movie),'command':command,
        'encoded_frames':len(image_paths),'timing_manifest':manifest.name,'timing_manifest_sha256':sha_file(manifest),
        'encode_seconds':encode_seconds,'encode_seconds_per_actual_frame':encode_seconds/len(image_paths),
        'decode_verification_seconds':decode_verification_seconds,
        'encoding_source_images':source_images,'nominal_codec_rate_hint_is_not_playback_cadence':nominal_rate,
        'encoding_verification':verification,'variable_movie_source_sha256':sha_file(Path(__file__))}
