"""Exact native skin transfer from the connected bilateral FEM, at 1x motion."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R.parent/'human-reference-pipeline-20261009/src'))
sys.path.insert(0,str(R/'original-renderer'))
from clay_render_soft import render_mesh_soft

p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
head=R.parent/'human-reference-pipeline-20261009/output/native-connected-head-study/shared-head-v16/head.npz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=np.load(a.case/'rest-native/native_motion.npz');loaded=np.load(a.case/'right-02-native/native_motion.npz')
assert str(base['source_head_sha256'])==sha(head)==str(loaded['source_head_sha256'])
assert np.array_equal(base['vertex_ids'],loaded['vertex_ids'])
h=np.load(head);v=h['vertices'].astype(np.float64);f=h['triangles'];ids=base['vertex_ids'];x0=v.copy();x1=v.copy();x0[ids]+=base['displacement'];x1[ids]+=loaded['displacement']
delta=loaded['displacement']-base['displacement'];mag=np.linalg.norm(delta,axis=1)
b=np.load(R/'output/bilateral-midface-fitted-v2/native-binding-v1/native_surface_binding.npz')
assert np.array_equal(ids,b['vertex_ids'])
report={'head_sha256':sha(head),'rest_motion_sha256':sha(a.case/'rest-native/native_motion.npz'),'loaded_motion_sha256':sha(a.case/'right-02-native/native_motion.npz'),'native_transfer_vertices':len(ids),'maximum_native_motion_design_units':float(mag.max()),'regions':{},'unit_note':'Authored design units, not calibrated physical millimetres','scope':'One-way original affine transfer; face orientation audit is not a self-contact proof'}
for name,mask in [('right_midface',(b['regions']==0)&(v[ids,0]>0)),('left_midface',(b['regions']==0)&(v[ids,0]<0)),('passive_oral_exterior',b['regions']==1),('passive_oral_mucosa',b['regions']==2)]:
    q=np.flatnonzero(mask);i=q[np.argmax(mag[q])];report['regions'][name]={'vertices':len(q),'maximum_displacement':float(mag[i]),'maximum_vertex_id':int(ids[i]),'vector_at_maximum_xyz':delta[i].tolist(),'mean_displacement':float(mag[q].mean())}
areas=[];cos=[]
for start in range(0,len(f),50000):
    ff=f[start:start+50000];p0=x0[ff];p1=x1[ff];n0=np.cross(p0[:,1]-p0[:,0],p0[:,2]-p0[:,0]);n1=np.cross(p1[:,1]-p1[:,0],p1[:,2]-p1[:,0]);l0=np.linalg.norm(n0,axis=1);l1=np.linalg.norm(n1,axis=1)
    assert np.all(l0>0) and np.all(l1>0)
    areas.extend(l1/l0);cos.extend(np.sum(n0*n1,axis=1)/(l0*l1))
report['all_head_triangles']={'count':len(f),'minimum_area_ratio':float(min(areas)),'maximum_area_ratio':float(max(areas)),'minimum_rest_loaded_normal_cosine':float(min(cos))}
assert min(cos)>0 and min(areas)>.8
a.output.mkdir(parents=True,exist_ok=False);(a.output/'native_response_report.json').write_text(json.dumps(report,indent=2)+'\n')
size=460;bg=(239,238,234);ink=(40,40,37);font=ImageFont.load_default(size=19);small=ImageFont.load_default(size=15)
sheet=Image.new('RGB',(2*size,2*size+188),bg);d=ImageDraw.Draw(sheet)
d.text((14,10),'Connected bilateral tissue drives the exact native skin',font=font,fill=ink)
d.text((14,38),'Left: undriven rest     Right: 2% right-cheek command, left cheek and mouth passive',font=small,fill=ink)
for row,(angle,framing) in enumerate([(0,{'center':[-4,-45.4,36],'span':323.9}),(-35,{'center':[0,-26,68],'span':145})]):
    for col,points in enumerate([x0,x1]):
        im=render_mesh_soft(points,f,np.zeros(len(f),np.int32),angle=angle,size=size,supersample=1,samples=10,shadow_resolution=640,material='gray',framing=framing)
        im.save(a.output/f'actual_native_{row}_{col}.png');sheet.paste(im,(col*size,68+row*size))
y=68+2*size
d.text((14,y+6),f'Actual motion at 1x. Maximum skin response {mag.max():.4f} authored units; the visible change is subtle.',font=small,fill=ink)
d.text((14,y+29),f'Passive mouth exterior max {report["regions"]["passive_oral_exterior"]["maximum_displacement"]:.5f}; undriven left midface max {report["regions"]["left_midface"]["maximum_displacement"]:.5f}.',font=small,fill=ink)
d.text((14,y+52),'All accepted tissue-boundary paths clear static bones; approximate nodal contact forces remain.',font=small,fill=ink)
d.text((14,y+75),'Finite origin gaps remain. Uncalibrated regional mechanics; dense skin self-contact is not certified.',font=small,fill=ink)
sheet.save(a.output/'actual_bilateral_native_skin_response.png');print(json.dumps(report,indent=2));print(a.output/'actual_bilateral_native_skin_response.png',flush=True)
