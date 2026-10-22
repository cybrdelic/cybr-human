"""Matched-time active/passive rendering, from hash-bound accepted states only.

No registration, fitting, motion scaling, time interpolation or looping. The
metrics distinguish real 3-D skin displacement from its normal-scale projection.
Visible-vertex metrics use vertices of front-view exterior triangles owning at
least one raster pixel after bone occlusion; they are not a visibility proof for
every included vertex. Pixel differences include shading and occlusion changes.
"""
from platform_runtime import resolve_tool
import argparse
import copy
import json
from pathlib import Path
from renderer_report_binding import recheck_report_bindings
import subprocess
import numpy as np
from PIL import Image, ImageDraw
from literal_renderer import load_geometry, Scene, project, sha_file, sha_array, font
from accepted_frames import load_accepted_frames, boundary_frame
from render_actual import movie_timing, verify_encoded_movie


def summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {'count': 0, 'maximum': None, 'rms': None, 'median': None, 'p95': None}
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite motion metric')
    return {'count': int(values.size), 'maximum': float(values.max()),
            'rms': float(np.sqrt(np.mean(values**2))),
            'median': float(np.median(values)), 'p95': float(np.percentile(values, 95))}


def match_times(active_times, passive_times):
    # Exact matching is intentional: nearest-time pairing would add an
    # undocumented approximation to the physical comparison.
    times, ia, ip = np.intersect1d(active_times, passive_times, return_indices=True)
    if not len(times):
        raise ValueError('No exactly matched accepted physical times')
    return times, ia, ip


def visible_front_vertices(scene, vertices):
    camera = scene.cameras[0]
    buffers = scene.raster.buffers(camera)
    scene.raster.draw(buffers, camera.check_boundary(vertices), scene.g['triangles'],
                      scene.g['boundary_class'], camera)
    scene.raster.draw(buffers, scene.bone_projections[0], scene.bone_faces,
                      scene.bone_classes, camera, palette=scene.bone_palette,
                      depth_cue=False, owner_offset=len(scene.g['triangles']))
    owners = np.unique(buffers[2])
    owners = owners[owners < len(scene.g['triangles'])]
    owners = owners[scene.g['boundary_class'][owners] == 0]
    return np.unique(scene.g['triangles'][owners]), int(len(owners))


def motion_metrics(scene, active_vertices, passive_vertices, active_image, passive_image,
                   front_panel_box=(45,180,745,970)):
    exterior = np.unique(scene.g['triangles'][scene.g['boundary_class'] == 0])
    displacement = active_vertices - passive_vertices
    visible_a, faces_a = visible_front_vertices(scene, active_vertices)
    visible_p, faces_p = visible_front_vertices(scene, passive_vertices)
    visible = np.union1d(visible_a, visible_p)
    projections = {}
    for name, camera in zip(('front', 'oblique_32_degrees', 'profile'), scene.cameras):
        projected = project(displacement, camera.degrees)
        pixel_distance = np.linalg.norm(projected[:, :2], axis=1)*camera.pixels_per_unit
        projections[name] = {'all_exterior_vertices_pixels': summary(pixel_distance[exterior])}
        if name == 'front':
            front_values = pixel_distance[visible]
            projections[name].update(
                visible_triangle_vertices_pixels=summary(front_values),
                visible_triangle_vertices_at_least_one_pixel=int(np.count_nonzero(front_values >= 1.)),
                visible_triangle_vertices_at_least_one_tenth_pixel=int(np.count_nonzero(front_values >= .1)))
    # This is the renderer's normal full-resolution front panel, excluding
    # text. No contrast enhancement or image registration is used.
    left,top,right,bottom=front_panel_box
    a = np.asarray(active_image)[top:bottom,left:right].astype(np.int16)
    p = np.asarray(passive_image)[top:bottom,left:right].astype(np.int16)
    if a.shape!=p.shape or a.shape!=(bottom-top,right-left,3):
        raise ValueError('Comparison front panel must match both literal images')
    rgb_difference = np.abs(a-p)
    return {
        'active_minus_passive_exterior_vertex_displacement_mm': summary(np.linalg.norm(displacement[exterior], axis=1)),
        'projected_displacement': projections,
        'front_visible_exterior_faces': {'active': faces_a, 'passive': faces_p},
        'front_visibility_metric_definition': 'Union of vertices of class-0 exterior triangles owning at least one front-view pixel after native-bone occlusion. Not every included vertex is necessarily visible.',
        'front_normal_scale_pixel_difference': {
            'panel_box_pixels_left_top_right_bottom':list(front_panel_box),
            'panel_pixels': int(a.shape[0]*a.shape[1]),
            'pixels_with_any_rgb_difference': int(np.any(rgb_difference > 0, axis=2).sum()),
            'pixels_with_rgb_difference_over_5': int(np.any(rgb_difference > 5, axis=2).sum()),
            'mean_absolute_rgb_difference': float(rgb_difference.mean()),
            'maximum_rgb_difference': int(rgb_difference.max()),
            'includes_lighting_and_occlusion_changes': True},
        'displacement_gain': 1., 'registration_or_fitting': False}


def comparison_image(active, passive, time_s, active_case, passive_case, playback_note, scale):
    out = Image.new('RGB', (2250, 2140), '#f4f6f8')
    draw = ImageDraw.Draw(out)
    def text(y, value, size=23, color='#354551'):
        draw.text((1125, y), value, font=font(size), fill=color, anchor='mt')
    text(22, 'CYBR HUMAN / matched-time actual accepted states', 34, '#20313d')
    text(75, f'Physical t = {time_s:.9g} s | identical cameras and literal 1x coordinates')
    if playback_note:
        text(113, playback_note, 18)
    text(155, passive_case, 25)
    out.paste(passive.crop((0, 130, 2250, 975)), (0, 195))
    text(1065, active_case, 25)
    out.paste(active.crop((0, 130, 2250, 975)), (0, 1105))
    text(1980, 'Coarse physical-site contact; approximate complementarity; NOT SURFACE-COLLISION CERTIFIED', 21, '#943c35')
    text(2020, 'No shape target, fitting, smoothing, displacement gain, interpolated frames or looping.', 21)
    text(2060, f'Engineering scale assumption: 1 atlas L0 = {scale:g} mm; not measured calibration.', 20)
    return out


def comparable_contact_provenance(provenance):
    """Ignore only independently measured setup wall times, never physical data."""
    comparable=copy.deepcopy(provenance)
    for path in (('setup_seconds',),('initial_site_state','site_query_seconds'),('rigid_query','build_seconds')):
        target=comparable
        for key in path[:-1]:
            target=target.get(key,{})
        target.pop(path[-1],None)
    return comparable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--active-frames', type=Path, required=True)
    parser.add_argument('--active-report', type=Path, required=True)
    parser.add_argument('--active-closed-path-audit', type=Path, required=True)
    parser.add_argument('--active-case', required=True)
    parser.add_argument('--passive-frames', type=Path, required=True)
    parser.add_argument('--passive-report', type=Path, required=True)
    parser.add_argument('--passive-closed-path-audit', type=Path, required=True)
    parser.add_argument('--passive-case', default='passive')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--movie', action='store_true')
    parser.add_argument('--playback-fps', type=float, default=24.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory to preserve evidence')
    g = load_geometry(args.geometry)
    active, aa = load_accepted_frames(args.active_frames, args.active_report, args.active_case, g,args.active_closed_path_audit)
    passive, pa = load_accepted_frames(args.passive_frames, args.passive_report, args.passive_case, g,args.passive_closed_path_audit)
    if not np.array_equal(active['source_initial_positions_mm'], passive['source_initial_positions_mm']):
        raise ValueError('Comparison requires exactly identical initial positions')
    if float(active['scale_mm_per_L0']) != float(passive['scale_mm_per_L0']):
        raise ValueError('Comparison requires identical physical scale')
    for key in ('source_phase_sha256', 'contact_mode', 'contact_enforcement', 'site_resolution_mm',
                'exterior_contact_stationarity','epigraph_mechanics_source_sha256','coarse_contact_provenance'):
        a_value,p_value=aa.get(key),pa.get(key)
        if key=='coarse_contact_provenance':
            a_value,p_value=map(comparable_contact_provenance,(a_value,p_value))
        if a_value != p_value:
            raise ValueError(f'Comparison cases have different {key}')
    if aa['contact_mode'] != 'coarse_physical_sites' or aa['contact_enforcement'] != 'compact_cap_barrier_v1':
        raise ValueError('This comparison caption currently supports only explicitly coarse compact-barrier contact')
    times, ia, ip = match_times(active['physical_times_s'], passive['physical_times_s'])
    speed = None
    note = ''
    if args.movie:
        combined = np.concatenate((active['positions_mm'][ia], passive['positions_mm'][ip]), axis=1)
        speed = movie_timing(times, combined, args.playback_fps)
        note = f'{args.playback_fps:g} fps playback; {speed:.9g} physical seconds per screen second; no interpolation'
    scene = Scene(g, scale_mm_per_L0=float(active['scale_mm_per_L0']))
    for frames, indices in ((active, ia), (passive, ip)):
        for index in indices:
            vertices = boundary_frame(frames, index, g)
            for camera in scene.cameras:
                camera.check_boundary(vertices)
    args.output.mkdir(parents=True)
    audit = {'active_input': aa, 'passive_input': pa, 'scene': scene.metadata(),
             'case_identity_ignores_only_setup_wall_time_fields':['setup_seconds',
                 'initial_site_state.site_query_seconds','rigid_query.build_seconds'],
             'exactly_matched_physical_times_s': times.tolist(), 'outputs': [], 'movie': None,
             'motion_metric_units': 'World distances in mm; projected distances at the saved normal camera pixel scale.'}
    for i, (t, ai, pi) in enumerate(zip(times, ia, ip)):
        av, pv = boundary_frame(active, ai, g), boundary_frame(passive, pi, g)
        kw = dict(reference_only=False, physical_time_s=float(t), playback_note=note,
                  contact_mode=aa['contact_mode'], site_resolution_mm=aa['site_resolution_mm'],
                  contact_enforcement=aa['contact_enforcement'])
        aimage, astats = scene.render(av, case=args.active_case, **kw)
        pimage, pstats = scene.render(pv, case=args.passive_case, **kw)
        metrics = motion_metrics(scene, av, pv, aimage, pimage)
        path = args.output/f'comparison-{i:06d}.png'
        comparison_image(aimage, pimage, float(t), args.active_case, args.passive_case,
                         note, scene.scale).save(path)
        audit['outputs'].append({'image': path.name, 'sha256': sha_file(path),
            'physical_time_s': float(t), 'active_source_frame_index': int(ai),
            'passive_source_frame_index': int(pi), 'active_boundary_sha256': sha_array(av),
            'passive_boundary_sha256': sha_array(pv), 'metrics': metrics,
            'render': {'active': astats, 'passive': pstats}})
        print(json.dumps({'image': str(path.resolve()), 'physical_time_s': float(t),
                          'motion_mm': metrics['active_minus_passive_exterior_vertex_displacement_mm'],
                          'front_projected': metrics['projected_displacement']['front']}), flush=True)
    if args.movie:
        movie = args.output/'matched-accepted-motion.mp4'
        command = [resolve_tool('ffmpeg'), '-nostdin', '-v', 'error', '-framerate', str(args.playback_fps),
                   '-i', str(args.output/'comparison-%06d.png'), '-frames:v', str(len(times)),
                   '-fps_mode', 'passthrough', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(movie)]
        subprocess.run(command, check=True)
        audit['movie'] = {'file': movie.name, 'sha256': sha_file(movie), 'command': command,
                          'encoded_frames': len(times), 'playback_fps': args.playback_fps,
                          'physical_seconds_per_screen_second': speed,
                          'frame_interpolation': False, 'looping': False,
                          'physical_first_and_last_sample_s':[float(times[0]),float(times[-1])],
                          'encoding_verification':verify_encoded_movie(movie,len(times),args.playback_fps,(2250,2140))}
    for path, expected in ((args.geometry, g['sha256']), (args.active_frames, aa['frame_sha256']),
                           (args.active_report, aa['report_sha256']), (args.passive_frames, pa['frame_sha256']),
                           (args.passive_report, pa['report_sha256']),
                           (args.active_closed_path_audit,aa['external_closed_site_path_audit']['audit_sha256']),
                           (args.passive_closed_path_audit,pa['external_closed_site_path_audit']['audit_sha256'])):
        if sha_file(path) != expected:
            raise RuntimeError('Input changed during rendering; outputs not qualified')
    recheck_report_bindings(aa);recheck_report_bindings(pa)
    (args.output/'comparison-audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
