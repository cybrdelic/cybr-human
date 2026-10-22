"""Only render the existing UNSOLVED reference. Never makes an animation."""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from literal_renderer import HERE, BASE, load_geometry, Scene, sha_file, sha_array

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',type=Path,default=BASE/'whole-face-continuum-staging/geometry/whole-face-continuum-final.npz')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.repeats<=20: parser.error('repeats must be 1 to 20')
    out=HERE/'reference-only-benchmark.png'; report_path=HERE/'reference-only-benchmark.json'
    started=time.perf_counter(); g=load_geometry(args.geometry); scene=Scene(g)
    preparation=time.perf_counter()-started
    records=[]
    for _ in range(args.repeats):
        image,record=scene.render(g['vertices'],reference_only=True); records.append(record)
    save_start=time.perf_counter(); image.save(out); save_seconds=time.perf_counter()-save_start
    assert sha_file(args.geometry)==g['sha256']
    result={'status':'UNSOLVED REFERENCE ONLY; no accepted frames; no animation generated',
        'scene':scene.metadata(),'geometry':g['path'],'reference_vertex_sha256':sha_array(g['vertices']),
        'triangle_sha256':sha_array(g['triangles']), 'mapping_parent_sha256':sha_array(g['exterior_tet']),
        'mapping_barycentric_sha256':sha_array(g['exterior_barycentric']),
        'rest_mapping_max_abs_error_L0':g['rest_mapping_max_abs_error'],
        'preparation_seconds':preparation,'repeated_three_view_seconds':[r['seconds'] for r in records],
        'median_three_view_seconds':statistics.median(r['seconds'] for r in records),
        'png_write_seconds':save_seconds,'output':str(out),'output_sha256':sha_file(out),
        'pixels':[2250,1200],'last_render_stats':records[-1]['views'],
        'rasterizer_source_sha256':sha_file(HERE/'rasterizer.cpp'),
        'rasterizer_library_sha256':sha_file(HERE/'rasterizer.so'),
        'python_source_sha256':sha_file(HERE/'literal_renderer.py'),
        'physics_solves':0,'geometric_coordinate_changes':0}
    report_path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
