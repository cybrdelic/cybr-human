"""Explicit shared nasal-root / medial orbital control surface.

The source eye-height bridge-to-canthus drop is approximately 21 units.
This editable symmetric cubic chart reduces that drop through both supports;
complete finite orbital patches are rebuilt against it, not held as islands.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
X=np.array([0,4,8,12,16,20,26.])
Y=np.array([-12,0,12,24,36,52.])
Z=np.array([
 [131.5,130.5,127.,121.,117.,114.,110.],
 [124.5,123.5,121.,117.,113.8,111.7,109.5],
 [117.4,116.8,114.8,111.,106.8,104.2,105.5],
 [117.,116.8,116.1,115.,113.2,111.3,108.9],
 [116.6,116.2,115.7,114.9,114.1,112.9,111.4],
 [112.9,112.8,112.6,112.4,111.9,111.2,109.8]])
def smooth(t):
 t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))
def rebuild_medial_support(p):
 xx,yy,zz=p.T;ax=np.abs(xx)
 # Mirrored controls impose a smooth bilateral central derivative.
 xs=np.r_[-X[:0:-1],X];zs=np.c_[Z[:,:0:-1],Z]
 carrier=smooth((26-ax)/10)*smooth((yy+12)/12)*smooth((32-yy)/10)*smooth((zz-88)/10)
 target=zz.copy();active=np.flatnonzero(carrier>0)
 # Shape-preserving interpolation prevents a cubic overshoot plateau above
 # the brow. Vectorized paired evaluation keeps memory bounded.
 ycurve=PchipInterpolator(Y,zs,axis=0)
 for start in range(0,len(active),4096):
  ids=active[start:start+4096];rows=ycurve(yy[ids]);curve=PchipInterpolator(xs,rows,axis=1);j=np.clip(np.searchsorted(xs,xx[ids])-1,0,len(xs)-2);k=np.arange(len(ids));dx=xx[ids]-xs[j];cc=curve.c
  target[ids]=((cc[0,j,k]*dx+cc[1,j,k])*dx+cc[2,j,k])*dx+cc[3,j,k]
 out=p.copy();out[:,2]=zz+carrier*(target-zz)
 return out,{'representation':'Symmetric editable 6 by 7 shape-preserving medial nasal-root/orbital control surface, terminated below the existing forehead and with finite orbital reconstruction on its actual boundary','x_controls':X.tolist(),'y_controls':Y.tolist(),'anterior_z_controls':Z.tolist(),'modified_vertices':int((carrier>0).sum()),'maximum_abs_depth_change':float(np.max(abs(out[:,2]-zz))),'source_eye_height_bridge_to_medial_canthus_depth_drop':21.0,'design':'Reduce the deep medial valley through both bridge recession and anterior canthal support; outer globes remain exact','limits':['Artist-authored relative depths, not anatomical measurement','Full changed surface, optical cap, finite return and visual tests required']}
