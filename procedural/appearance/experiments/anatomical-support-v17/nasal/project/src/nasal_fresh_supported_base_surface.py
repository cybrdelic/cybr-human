"""Direct supported basal sections without an exterior pre-carved nostril basin.

Forks the editable 17x17 fresh guide, including its fitted outer attachment
coefficients. New named 3D section controls carry the paired tip down over the
basal return and broaden its medial/alar support. This is one cubic control
surface, not a displacement sampled onto an inherited interior mesh.
The v43 control source, fit and insertion cache remain unchanged.
"""
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from nasal_fresh_basal_control_net import B,G,CONTROL as FRESH_CONTROL,GUIDE_Y,GUIDE_Z,BOUNDARY,width,raw_surface as original_raw,evaluate as original_exact,OUT as ROOT_OUT
from nasal_fresh_tangent_surface import FREE,coefficients,derivatives

OUT=ROOT_OUT/'supported-basal-control-net-v4'

# Physical control-row offsets, positive half from axis through paired dome,
# medial support, inlet roof, alar body, cheek return, sidewall and boundary.
# Outer two layers and the attachment derivative controls are unmodified.
DOWNWARD_ROLL=np.array([
 [0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0],
 [-.3,-.4,-.2,0,.1,.05,0,0,0],
 [-1.1,-1.5,-1.6,-.6,-.3,-.1,0,0,0],
 [-2.4,-2.7,-2.8,-2.4,-1.2,-.2,0,0,0],
 [-2.2,-2.4,-2.6,-2.3,-1.15,-.2,0,0,0],
 [-1.3,-1.4,-1.4,-1.1,-.5,-.05,0,0,0],
 [-.3,-.35,-.3,-.2,-.1,0,0,0,0],
 [0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0],
],float)
# Absolute original section targets. The former inlet-support wells are
# replaced by a gradual basal arch. Only the finite inlet has deep recession.
SUPPORTED_BASAL_Z=np.array([
 [121.1,121.3,121.,119.,117.,114.,111.,109.5,108.5],
 [121.,121.1,121.,120.7,118.7,115.4,111.6,109.5,108.8],
 [125.9,124.1,124.2,122.9,121.7,117.5,112.7,111.,110.7],
 [133.9,131.2,128.4,127.,127.2,121.3,115.3,112.3,112.2],
 [137.3,138.8,134.3,131.,130.1,124.1,117.7,114.5,113.3],
 [137.3,140.1,137.7,132.5,126.9,122.,118.1,114.8,113.6],
 [135.3,136.,134.,129.,124.7,120.,117.5,114.5,113.8],
 [133.2,133.,130.,126.,121.2,117.9,115.7,114.2,113.6],
 [128.,127.5,125.5,121.,117.8,114.6,112.7,111.5,112.],
 [122.4,122.,120.5,117.5,114.5,112.7,111.,109.5,108.],
 [118.8,118.5,117.6,115.7,113.7,111.8,110.5,109.5,108.6],
],float)



def make_controls():
    control=FRESH_CONTROL.copy()
    control[FREE[:,0],FREE[:,1],2]+=coefficients()
    y=-40+55*G
    half=PchipInterpolator(GUIDE_Y,DOWNWARD_ROLL,axis=0)(y)
    control[:,:,1]+=np.c_[half[:,:0:-1],half]
    half=PchipInterpolator(GUIDE_Y,SUPPORTED_BASAL_Z,axis=0)(y)-PchipInterpolator(GUIDE_Y,GUIDE_Z,axis=0)(y)
    control[:,:,2]+=np.c_[half[:,:0:-1],half]
    return control


CONTROL=make_controls()


def evaluate(uv):
    uv=np.asarray(uv,float)
    raw=np.einsum('ni,ijc,nj->nc',B(uv[:,1]),CONTROL,B(uv[:,0]),optimize=True)
    # The small exact outer-curve residual is identical because all edge
    # control points and their first inward derivative layers stay fixed.
    return raw+original_exact(uv)-original_raw(uv)


def build_preview():
    uv=np.stack(np.meshgrid(np.linspace(0,1,65),np.linspace(0,1,81),indexing='ij'),-1).reshape(-1,2)
    vertices=evaluate(uv);ind=np.arange(len(uv)).reshape(65,81)
    a=ind[:-1,:-1].ravel();b=ind[1:,:-1].ravel();c=ind[1:,1:].ravel();d=ind[:-1,1:].ravel()
    triangles=np.r_[np.c_[a,b,c],np.c_[a,c,d]]
    p=vertices[triangles];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1)*.5
    if area.min()<1e-10:raise ValueError('Invalid new control surface')
    boundary_uv=np.c_[(BOUNDARY[:,0]/width(BOUNDARY[:,1])+1)/2,(BOUNDARY[:,1]+40)/55]
    error=float(np.linalg.norm(evaluate(boundary_uv)-BOUNDARY,axis=1).max())
    if error>1e-9:raise ValueError('Real attachment changed')
    report={'method':'Direct 3D section-control revision: lower supported paired tip, continuous basal arch without exterior inlet wells, preserving fixed outer control layers',
        'vertices':len(vertices),'triangles':len(triangles),'minimum_triangle_area':float(area.min()),
        'outer_attachment_maximum_error':error,'minimum_exterior_front_projected_double_area':float(cross[:,2].min()),
        'downward_roll_control_offsets':DOWNWARD_ROLL.tolist(),'absolute_supported_basal_depth_controls':SUPPORTED_BASAL_Z.tolist(),
        'visual_gate_passed':None,'approved_for_integration':False,'approved_for_selection':False}
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'solid_exterior.npz',vertices=vertices,triangles=triangles,uv=uv,control_net=CONTROL)
    (OUT/'solid_exterior_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':print(json.dumps(build_preview(),indent=2),flush=True)
