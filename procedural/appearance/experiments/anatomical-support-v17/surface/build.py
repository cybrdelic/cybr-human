"""Independent whole-form study: cranial planes, coherent midface, finite lids.

The exact preferred head is retained elsewhere. This candidate deliberately
changes broader proportions and regenerates coupled eyelid sections.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from orbital_surface import build
import orbital_surface as orbital_module
from refresh_anatomical_semantics import refresh

import argparse,shutil
parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-variant',action='store_true');a=parser.parse_args()
PACKAGE=Path(__file__).resolve().parent
OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True);(OUT/'interfaces').mkdir(exist_ok=True);shutil.copy2(PACKAGE/'MIDFACE_CONTROLS.json',OUT/'MIDFACE_CONTROLS.json');orbital_module.OUT=OUT
ROOT=OUT.parents[1]
BASE=PACKAGE/'dependencies/v22'
MARGIN=PACKAGE/'dependencies/margin-v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.baseline)=='c47053b8b521a6d85f62f9c38a1bbe1b2243daa228d93d59911fb9ba544e2d34'
import shutil
for side in ['right','left']:shutil.copy2(MARGIN/'interfaces'/f'{side}_cut.npz',OUT/'interfaces'/f'{side}_cut.npz')
h=np.load(a.baseline);v=h['vertices'].copy();f=h['triangles'];oid=h['object_ids'];original=v.copy()
contract=np.load(PACKAGE/'anatomy_contract.npz');orbital=[];orbit_reports={}
(OUT/'local-surfaces').mkdir(exist_ok=True)
def original_orbital(side):
 local=contract[side+'_local_to_v22_vertex_ids'];face_ids=contract[side+'_v22_face_ids'];cut=np.load(MARGIN/'interfaces'/f'{side}_cut.npz');c=np.array([31.3 if side=='right' else -31.3,12,94.6]);sign=1 if side=='right' else -1;d=cut['boundary_points']-c;order=np.argsort(np.mod(np.arctan2(d[:,1],sign*d[:,0]),2*np.pi));reverse=np.full(len(original),-1,np.int64);reverse[local]=np.arange(len(local));tri=reverse[h['triangles'][face_ids]];assert np.all(tri>=0)
 return {'vertices':original[local].copy(),'triangles':tri,'host_boundary_vertex_ids':np.arange(len(order)),'boundary_source_order':order}

for side,sign in [('right',1),('left',-1)]:
 old=original_orbital(side)
 local=contract[side+'_local_to_v22_vertex_ids']
 assert np.all(local>=0) and np.array_equal(v[local],old['vertices'])
 new=build(side,sign);assert np.array_equal(old['triangles'],new['triangles'])
 assert np.array_equal(old['vertices'][old['host_boundary_vertex_ids']],new['vertices'][new['host_boundary_vertex_ids']])
 v[local]=new['vertices'];orbital.extend(local.tolist())
 np.savez_compressed(OUT/'local-surfaces'/f'{side}_orbital.npz',**new,local_to_global_vertex_ids=local)
 orbit_reports[side]={'original_central_upper_half_height':4.65,'new_central_upper_half_height':5.8,'original_central_lower_half_height':4.35,'new_central_lower_half_height':4.35,'wetline_cornea_support':'Smooth extra radial support up to 0.25 authored units, fading into posterior socket','nasal_canthal_anterior_lift':2.6,'hidden_return':'Continuous transition from actual wetline radial distance to the original closed socket radius','outer_attachment_exact':True,'same_topology':True}

def smooth(x):
 x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))

def anatomical_field(p):
 """One composable low-frequency 3-D map, not mesh smoothing or texture."""
 x,y,z=p.T;ax=np.abs(x);d=np.zeros_like(p)
 # Approximate reference ratios already support the overall head height.
 # Sculpt temporal/frontal planes rather than uniformly shrinking the skull.
 vault=smooth((y-28)/74)
 d[:,1]-=4.0*vault
 d[:,0]-=.042*x*vault
 front=smooth((z-20)/62)
 d[:,2]-=3.0*smooth((y-34)/38)*front*np.exp(-(x/58)**4)
 temple=np.exp(-((y-34)/25)**2)*smooth((ax-43)/13)*smooth((77-ax)/14)*smooth((z+12)/45)
 d[:,0]-=np.sign(x)*2.8*temple
 # Exact protected ocular support for broad cranial/temporal map.
 dist=np.minimum(np.linalg.norm(p-[31.3,12,94.6],axis=1),np.linalg.norm(p-[-31.3,12,94.6],axis=1))
 safe=smooth((dist-17.7)/8)
 d*=safe[:,None]
 return d

skin=np.unique(f[oid==0]);delta=anatomical_field(v[skin]);v[skin]+=delta
from coarse_midface_surface import sample_supported_surface
v[skin],midface_report=sample_supported_surface(v[skin],OUT/'MIDFACE_CONTROLS.json')

from medial_orbital_support import rebuild_medial_support
v[skin],medial_report=rebuild_medial_support(v[skin])

# Reconstruct the complete orbital chart against the new coarse host.

for side,sign in [('right',1),('left',-1)]:
 old=original_orbital(side);oldmap=contract[side+'_local_to_v22_vertex_ids'];nb=len(old['host_boundary_vertex_ids']);order=old['boundary_source_order'];path=OUT/'interfaces'/f'{side}_cut.npz';cut=dict(np.load(path));boundary_global=oldmap[old['host_boundary_vertex_ids']]
 cut['boundary_points'][order]=v[boundary_global]
 # Area-weighted normals from retained exterior host only. The provisional old orbital patch must not contaminate the new attachment tangents.
 flag=np.zeros(len(v),bool);flag[boundary_global]=True;current_patch_faces=contract[side+'_v22_face_ids'];patchmask=np.zeros(len(f),bool);patchmask[current_patch_faces]=True;incident=f[np.any(flag[f],axis=1)&~patchmask];tri=v[incident];nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=np.zeros_like(v)
 for j in range(3):np.add.at(normals,incident[:,j],nn)
 normal=normals[boundary_global];normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-15);cut['boundary_normals'][order]=normal;np.savez_compressed(path,**cut)
 new=build(side,sign);mapping=oldmap.copy();unsorted=np.empty(nb,np.int64);unsorted[order]=boundary_global;mapping[:nb]=unsorted[new['boundary_source_order']];assert len(mapping)==len(new['vertices'])
 v[mapping]=new['vertices'];face_ids=contract[side+'_v22_face_ids'];assert np.all(face_ids>=0);f[face_ids]=mapping[new['triangles']]
 np.savez_compressed(OUT/'local-surfaces'/f'{side}_orbital.npz',**new,local_to_global_vertex_ids=mapping)
 orbit_reports[side]['outer_attachment_exact']=False;orbit_reports[side]['attachment']='Whole finite patch regenerated on the new coarse host boundary; tangents use retained exterior host area-weighted normals, excluding the provisional old patch';orbit_reports[side]['same_topology']=bool(np.array_equal(new['triangles'],old['triangles']))
from oral_support import reshape_oral_support
v[skin],oral_report=reshape_oral_support(v[skin])
from mandibular_support import rebuild_mandibular_support
v[skin],mandibular_report=rebuild_mandibular_support(v[skin],OUT)

globe=np.unique(f[oid>0]);assert np.array_equal(v[globe],original[globe])

np.savez_compressed(OUT/'candidate_head.npz',vertices=v,triangles=f,object_ids=oid)
refresh(OUT/'candidate_head.npz',BASE/'anatomical_semantics/authored_lip_lid_semantics.npz',OUT/'anatomical_semantics')
changed=np.any(v!=original,axis=1);modified=np.flatnonzero(np.any(changed[f],axis=1)&(oid==0))
np.savez_compressed(OUT/'retained_source_maps.npz',modified_face_ids=modified,changed_vertex_ids=np.flatnonzero(changed),orbital_vertex_ids=np.unique(orbital))
minimum=np.inf
for start in range(0,len(f),30000):
 p=v[f[start:start+30000]];minimum=min(minimum,float(np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1).min()/2))
assert minimum>1e-12
# Sample the authored broad map's spatial derivative separately from lids.
grid=np.stack(np.meshgrid(np.linspace(-82,82,31),np.linspace(-84,110,37),np.linspace(-68,141,35),indexing='ij'),-1).reshape(-1,3)
jac=np.tile(np.eye(3),(len(grid),1,1));eps=.002
for axis in range(3):
 step=np.zeros(3);step[axis]=eps;jac[:,:,axis]+=(anatomical_field(grid+step)-anatomical_field(grid-step))/(2*eps)
det=np.linalg.det(jac);sv=np.linalg.svd(jac,compute_uv=False)[:,-1]
assert det.min()>0 and sv.min()>.35
report={'head_sha256':sha(OUT/'candidate_head.npz'),'base_head_sha256':sha(a.baseline),'vertices':len(v),'triangles':len(f),'same_topology':bool(np.array_equal(f,h['triangles'])),'same_vertex_count':True,'globes_bitwise_exact':True,'retained_baseline':'Exact preferred v53 remains available and unchanged','selected':False,'orbital_sections':orbit_reports,'changed_vertices':int(changed.sum()),'modified_skin_faces':len(modified),'coarse_midface':midface_report,'oral_support':oral_report,'medial_orbital_support':medial_report,'mandibular_support':mandibular_report,'maximum_motion':float(np.linalg.norm(v-original,axis=1).max()),'minimum_triangle_area':minimum,'broad_map_sampled_jacobian':{'points':len(grid),'minimum_determinant':float(det.min()),'minimum_singular_value':float(sv.min()),'scope':'Cranial/temporal volumetric map only. Midface is a separately regular parameterized support surface; no global injectivity claim'},'limitations':['New whole-form candidate; retained host is intentionally no longer bitwise exact','Geometry, optics and groom need fresh exact fit and intersection checks','Qualitative fictional reference; no scanned or imported head and no image texture']}
(OUT/'ASSEMBLY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if not a.allow_variant:assert sha(OUT/'candidate_head.npz')=='5fb99ed9a5e113cd77188339f851c1db40ba952aacf5afbecf5484cd583a4a9c','Fresh anatomy source differs from the captured head'
