"""Original editable 3D cheek, brow and independent orbital depth controls.

A frozen pair of regularized cubic cages is evaluated in sequence, followed
by an injective compact translation field around each independent globe.
Authoring coordinates are design units, not calibrated anatomy.
"""
import numpy as np
from anatomical_volume_cage import displacement as cage_displacement
CENTERS=np.array([[31.3,12.,92.1],[-31.3,12.,92.1]])
SHIFT=2.5;SUPPORT_FRACTION=.58

def smooth(t):
 t=np.clip(t,0.,1.);return t*t*t*(10+t*(-15+6*t))

def cheek_displacement(points,coef):
 p=np.asarray(points,float);d=cage_displacement(p,coef,protect_eye_interfaces=False,origin=np.array([-98.,-91.,-14.]),step=7.,shape=(29,21,25))
 dist=np.minimum(np.linalg.norm(p-CENTERS[0],axis=1),np.linalg.norm(p-CENTERS[1],axis=1))
 gate=smooth((dist-14.)/8.)*smooth((np.abs(p[:,0])-13)/10)*smooth((80-np.abs(p[:,0]))/10)*smooth((p[:,1]+73)/13)*smooth((22-p[:,1])/16)*smooth((p[:,2]-3)/18)
 return d*gate[:,None]

def support_displacement(points,coef):
 p=np.asarray(points);d=cage_displacement(p,coef,False,origin=np.array([-91.,-56.,35.]),step=8.,shape=(24,18,15))
 dist=np.minimum(np.linalg.norm(p-CENTERS[0],axis=1),np.linalg.norm(p-CENTERS[1],axis=1))
 gate=smooth((dist-14)/9)*smooth((abs(p[:,0])-12)/9)*smooth((72-abs(p[:,0]))/10)*smooth((p[:,1]+44)/12)*smooth((62-p[:,1])/15)*smooth((p[:,2]-45)/15)
 return d*gate[:,None]

def orbital_displacement(points,shift=SHIFT):
 if 1.875*abs(shift)/16>=1:raise ValueError('Orbital depth exceeds the monotone-field bound')
 if 60>=np.linalg.norm(CENTERS[0]-CENTERS[1]):raise ValueError('Orbital support spheres overlap')
 p=np.asarray(points);weight=np.zeros(len(p))
 for center in CENTERS:weight+=1-smooth((np.linalg.norm(p-center,axis=1)-14)/16)
 d=np.zeros_like(p,dtype=float);d[:,2]=shift*weight;return d

def build(vertices,triangles,object_ids,cheek_coefficients,support_coefficients):
 v=np.asarray(vertices);skin=np.unique(triangles[object_ids==0]);w=v.copy()
 w[skin]+=cheek_displacement(v[skin],cheek_coefficients)
 w[skin]+=SUPPORT_FRACTION*support_displacement(w[skin],support_coefficients)
 w=w+orbital_displacement(w)
 for obj in (1,2):
  ind=np.unique(triangles[object_ids==obj])
  if not np.array_equal(w[ind],v[ind]+[0,0,SHIFT]):raise ValueError('Independent globe is not the expected exact translation')
 # Test all changed triangles without retaining an all-mesh normal array.
 changed=np.any(w!=v,axis=1);minimum=1.;area_ratio=1.
 for start in range(0,len(triangles),100000):
  ff=triangles[start:start+100000];ff=ff[np.any(changed[ff],axis=1)]
  if len(ff)==0:continue
  a=v[ff];b=w[ff];n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);l0=np.linalg.norm(n0,axis=1);l1=np.linalg.norm(n1,axis=1)
  minimum=min(minimum,float((np.sum(n0*n1,axis=1)/np.maximum(l0*l1,1e-20)).min()));area_ratio=min(area_ratio,float((l1/np.maximum(l0,1e-20)).min()))
 if minimum<=0 or area_ratio<.1:raise ValueError('Discrete triangle fold or severe collapse')
 return w,{'original_centers':CENTERS.tolist(),'new_centers':(CENTERS+[0,0,SHIFT]).tolist(),'globe_radius':12.3,'orbital_shift':SHIFT,'support_fraction':SUPPORT_FRACTION,'minimum_changed_face_normal_cosine':minimum,'minimum_changed_face_area_ratio':area_ratio,'changed_vertices':int(changed.sum()),'topology_unchanged':True,'limits':'Discrete local triangle checks and exact rigid globes are verified; no global intersection or anatomy certificate.'}
