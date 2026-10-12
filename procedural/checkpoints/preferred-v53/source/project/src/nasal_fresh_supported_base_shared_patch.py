"""Exact existing v8 shared-boundary insertion cache for the fresh nose.

The approved fresh control surface is sampled directly. No old interior mesh,
collar, inlet chart or artistic correction is reused. The first N rows are the
unchanged actual attachment vertices, in the pinned contract order.
"""
import hashlib
import json
import numpy as np
from nasal_fresh_basal_control_net import CONTRACT,BOUNDARY
from nasal_fresh_supported_base_surface import OUT
from nasal_fresh_supported_base_inlets import build_preview
from stitch_anatomical_patch import stitch_patch


def build_patch():
    v,f,parts,uv,loops,report=build_preview(boundary_points=BOUNDARY,resolution=(97,113))
    if not np.array_equal(v[:len(BOUNDARY)],BOUNDARY):raise ValueError('Actual outer boundary was changed')
    with np.load(CONTRACT) as q:
        retained_v=q['retained_crop_vertices'];retained_f=q['retained_crop_triangles'];local_to_global=q['local_to_global']
        global_boundary=q['boundary_global_vertex_ids']
    lookup={int(g):i for i,g in enumerate(local_to_global)}
    local_boundary=np.array([lookup[int(g)] for g in global_boundary])
    _,_,_,stitch=stitch_patch(retained_v,retained_f,np.zeros(len(retained_f),np.uint8),local_boundary,v,f,tolerance=1e-12)
    report['bounded_actual_retained_crop_stitch']=stitch
    report['visual_gate_passed']=None
    report['approved_for_integration']=False
    report['approved_for_selection']=False
    report['approval_scope']='Control preparation and small-array construction only. New covered-tip local exterior/inlet and whole-head clay gates remain pending.'
    report['source_surface_changed_during_resampling']=False
    report['integration_api']='integrate_nasal_v8_shared.integrate(vertices,triangles,object_ids,patch_path=explicit_fitted_nasal_patch_path)'
    report['scope']='Exact shared boundary and checked bounded retained-crop insertion; no full-head mesh load or selected mesh edit.'
    path=OUT/'fitted_nasal_patch.npz'
    np.savez_compressed(path,vertices=v,triangles=f,face_parts=parts,exterior_uv=uv,root_vertex_ids=loops)
    report['patch_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'fitted_nasal_patch_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':print(json.dumps(build_patch(),indent=2),flush=True)
