"""Original 3D cubic B-spline anatomical deformation cage.

A regular volumetric cage, driven by named authored 3D curve landmarks, moves
skin in all three coordinates. This is not a front-depth graph or imported
head. Landmark changes are exploratory reference-guided design decisions.
"""
from pathlib import Path
import json,numpy as np
from scipy.sparse import coo_matrix,diags,kron,eye
from scipy.sparse.linalg import cg
ORIGIN=np.array([-128.,-208.,-144.]);STEP=16.;SHAPE=(17,23,21)

def basis(points,origin=ORIGIN,step=STEP,shape=SHAPE):
 u=(np.asarray(points)-np.asarray(origin))/step;k=np.floor(u).astype(int);t=u-k
 w=[np.stack([(1-a)**3/6,(3*a**3-6*a*a+4)/6,(-3*a**3+3*a*a+3*a+1)/6,a**3/6],axis=1) for a in t.T]
 inds=[];weights=[]
 for i in range(4):
  for j in range(4):
   for h in range(4):
    ix=np.clip(k[:,0]+i-1,0,shape[0]-1);iy=np.clip(k[:,1]+j-1,0,shape[1]-1);iz=np.clip(k[:,2]+h-1,0,shape[2]-1)
    inds.append((ix*shape[1]+iy)*shape[2]+iz);weights.append(w[0][:,i]*w[1][:,j]*w[2][:,h])
 return np.stack(inds,1),np.stack(weights,1)

def authored_controls():
 records=[]
 def add(name,p,d,weight=1.):records.append({'region':name,'source':list(p),'displacement':list(d),'weight':weight})
 def pair(name,p,d,weight=1.):
  add(name,p,d,weight)
  if p[0]:add(name,[-p[0],p[1],p[2]],[-d[0],d[1],d[2]],weight)
 # Each chain lives in three dimensions and has independently authored
 # lateral and depth motion. The nose tip projection is explicitly retained.
 for p,d in [((18,27,119),(0,1,-3.2)),((30,28,114),(0,1,-2.8)),((44,27,102),(1,1,-2.)),((51,20,87),(1,0,-1.))]:pair('brow_relief_curve',p,d,1.5)
 for p,d in [((31,-6,109),(1.6,.6,2.)),((43,-11,99),(2.5,.7,2.8)),((53,-15,82),(3.0,.5,2.1)),((62,-15,55),(2.2,0,0.))]:pair('malar_roll_curve',p,d,1.5)
 for p,d in [((32,-35,105),(1,1,1.)),((43,-38,83),(2.4,0,-2.)),((50,-45,66),(2.8,0,-1.2))]:pair('buccal_turn_curve',p,d)
 for p,d in [((49,-58,57),(2.4,0,-.5)),((51,-67,24),(2.2,0,0)),((40,-77,72),(2.0,0,-.5)),((26,-87,101),(1.4,0,-.5)),((13,-94,111),(.5,0,-1.))]:pair('mandibular_curve',p,d,1.2)
 for x,z in [(0,122),(7,122),(15,119),(25,108)]:pair('oral_seam_elevation',(x,-54.4,z),(0,3.7,0.7 if x<20 else 1.8),2.)
 for p,d in [((0,-44,125),(0,2.0,0)),((10,-46,124),(0,2.8,.5)),((0,-61,124),(0,3.5,1.2)),((13,-61,120),(0,3.3,1.4)),((23,-61,109),(.6,2.5,1.2)),((0,-72,119),(0,.5,-1.4)),((0,-85,123),(0,0,-1.2))]:pair('perioral_support_curve',p,d,1.4)
 for p,d in [((0,18,118),(0,0,-.7)),((0,0,129),(0,0,0)),((0,-14,138),(0,0,0)),((0,-22,141),(0,0,0)),((7,0,126),(1.3,0,-.4)),((9,-12,130),(1.0,0,-.6)),((13,-24,129),(.7,0,-.5)),((5,-24,138),(.5,0,.5)),((0,-33,127),(0,0,0))]:pair('nasal_plane_framework',p,d,1.8)
 for p,d in [((0,110,16),(0,-3.5,0)),((30,102,20),(-.7,-2.8,0)),((0,80,103),(0,-1.8,-1.3)),((36,77,91),(-.6,-1.5,-.8)),((0,43,119),(0,0,-.8)),((58,53,73),(-1.2,-.5,-.4)),((0,78,-52),(0,-1.5,1.0)),((41,56,-57),(-.5,-.8,.5))]:pair('cranial_envelope_curve',p,d)
 for p in [(0,-100,107),(0,-110,73),(0,-140,68),(45,-125,48),(70,-35,10),(70,15,10),(70,30,10),(0,10,-67),(43,7,-59),(0,-50,-20),(45,-65,-25),(70,-10,-15),(80,-25,0)]:pair('preserved_anchor',p,(0,0,0),2.)
 return records

def build_cage(controls=None,origin=ORIGIN,step=STEP,shape=SHAPE):
 rec=authored_controls() if controls is None else controls;p=np.array([r['source'] for r in rec]);target=np.array([r['displacement'] for r in rec]);weight=np.sqrt(np.array([r['weight'] for r in rec]));idx,w=basis(p,origin,step,shape);count=np.prod(shape)
 C=coo_matrix((w.ravel(),(np.repeat(np.arange(len(p)),64),idx.ravel())),shape=(len(p),count)).tocsr();C=diags(weight)@C;target=target*weight[:,None]
 ls=[]
 for n in shape:ls.append(diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1]).tocsr())
 L=kron(kron(ls[0],eye(shape[1])),eye(shape[2]))+kron(kron(eye(shape[0]),ls[1]),eye(shape[2]))+kron(kron(eye(shape[0]),eye(shape[1])),ls[2]);L=L.tocsr()
 grid=np.indices(shape).reshape(3,-1).T;fixed=np.any((grid<2)|(grid>=np.array(shape)-2),axis=1);free=np.flatnonzero(~fixed)
 Q=(C.T@C+.004*(L.T@L)+diags(np.full(count,.0003))).tocsr();A=Q[free][:,free];rhs=(C.T@target)[free];M=diags(1/np.maximum(A.diagonal(),1e-10));coef=np.zeros((count,3));states=[]
 for axis in range(3):
  sol,info=cg(A,rhs[:,axis],M=M,rtol=1e-9,maxiter=2000);coef[free,axis]=sol;states.append(int(info))
 if any(states):raise RuntimeError('Cage solve did not converge')
 achieved=C@coef;err=(achieved-target)/weight[:,None]
 return coef,{'method':'3D cubic B-spline control cage with anatomical landmark curves, second-difference regularization and fixed outer layers','origin':np.asarray(origin).tolist(),'spacing':step,'shape':list(shape),'controls':rec,'rms_landmark_displacement_error':float(np.sqrt(np.mean(err*err))),'max_landmark_displacement_error':float(np.linalg.norm(err,axis=1).max()),'solve_states':states,'units':'Approximate authored design units, not calibrated anatomy'}

def displacement(points,coef,protect_eye_interfaces=True,origin=ORIGIN,step=STEP,shape=SHAPE,eye_center_z=90.5):
 p=np.asarray(points);out=np.zeros_like(p,dtype=float)
 for start in range(0,len(p),10000):
  q=p[start:start+10000];idx,w=basis(q,origin,step,shape);d=np.einsum('nk,nkc->nc',w,coef[idx])
  if protect_eye_interfaces:
   # Preserve the complete independent-globe interface; broad brow/cheek
   # movement enters gradually beyond the finite eyelid/socket tissue.
   dist=np.minimum(np.linalg.norm(q-[31.3,12,eye_center_z],axis=1),np.linalg.norm(q-[-31.3,12,eye_center_z],axis=1));t=np.clip((dist-14.)/14.,0,1);gate=t**3*(10-15*t+6*t*t);d*=gate[:,None]
  out[start:start+len(q)]=d
 return out

def apply_to_mesh(vertices,object_ids,triangles,coef,eye_center_z=90.5):
 v=np.asarray(vertices).copy();skin=np.unique(triangles[object_ids==0]);v[skin]+=displacement(v[skin],coef,eye_center_z=eye_center_z)
 return v
