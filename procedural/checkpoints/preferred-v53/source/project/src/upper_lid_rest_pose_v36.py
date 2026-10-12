"""Coherent upper-lid resting pose on the independent original globes.

The selected opening exposes the complete upper iris cap. A restrained
angular movement carries finite upper lid and inner return tissue along the
same globe-centered coordinates, with broad orbital support. Lower margins,
canthi, globes and remote face are preserved by smooth local support. This is
an authored rest-pose study, not a complete blink rig or optical-fit proof.
"""
import numpy as np
from coordinated_facial_depth_v25 import smooth
CENTERS=np.array([[31.3,12.,94.6],[-31.3,12.,94.6]])
ANGLE=.115

def move_points(points):
 p=np.asarray(points);w=p.copy()
 for c in CENTERS:
  d=p-c;r=np.linalg.norm(d,axis=1);rho=np.linalg.norm(d[:,1:],axis=1)
  # The front-facing angular chart never reaches the back of the globe.
  span=smooth((12.5-abs(d[:,0]))/5.5)
  upper=smooth((d[:,1]-.6)/3.)*smooth((17.-d[:,1])/7.)
  depth=smooth((d[:,2]-3.)/6.)
  support=smooth((23.-r)/7.)
  a=ANGLE*span*upper*depth*support
  # Rotation about local X preserves globe-relative radius and thickness.
  co=np.cos(a);si=np.sin(a)
  w[:,1]+=d[:,1]*co-d[:,2]*si-d[:,1]
  w[:,2]+=d[:,1]*si+d[:,2]*co-d[:,2]
 return w

def deform(v,f,ids):
 w=v.copy();skin=np.unique(f[ids==0]);w[skin]=move_points(v[skin]);return w
