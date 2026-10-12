"""Authored lower-lid cushion blending into infraorbital cheek support.

This additive radial tissue field is a local shape study on the existing
connected skin, not a replacement topology or an optical-eye deformation.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from upper_lid_body_v7 import CENTERS,smooth,deform as upper_deform
DISTANCE=np.array([-2.,0.,1.,2.8,5.,8.,12.,16.])
RADIAL=np.array([0.,0.,.20,.95,1.15,.68,.12,0.])
PROFILE=PchipInterpolator(DISTANCE,RADIAL,extrapolate=False)

def lower_window_y(x):
 h=11.6;hu=5.9;hl=3.6;rounding=.16
 width=h/np.sqrt(1-2*rounding*np.log(2)/(hu+hl));k=1-(x/width)**2
 arc=rounding*np.arccosh(np.maximum(1.,.5*np.exp(np.clip((hu+hl)*k/(2*rounding),-40,40))))
 return .065*x+.5*(hu-hl)*k-arc

def displacement(points):
 p=np.asarray(points,float);d=np.zeros_like(p)
 for sign,c in zip((1.,-1.),CENTERS):
  rel=p-c;r=np.linalg.norm(rel,axis=1);x=sign*rel[:,0];y=rel[:,1];z=rel[:,2]
  # The broad lower body is placed by distance from the aperture curve.
  s=lower_window_y(x)-y
  profile=PROFILE(np.clip(s,DISTANCE[0],DISTANCE[-1]))
  # The inferior support is longer laterally and gently tapers at the canthi.
  lateral=smooth((16-np.abs(x))/6.)
  depth=smooth((z-1.)/5.)
  radial=smooth((r-12.8)/2.)*smooth((33-r)/8.)
  dr=profile*lateral*depth*radial
  d+=rel*(dr/np.maximum(r,1e-12))[:,None]
 return d

def deform(v,f,ids):
 w=upper_deform(v,f,ids);skin=np.unique(f[ids==0]);w[skin]+=displacement(w[skin]);return w
