"""Original three-dimensional upper-lid bearing and tissue-body study.

The existing connected finite lid/socket moves around an independent globe.
A meridional tissue curve authors the broad tarsal body and upper fold. This is
geometry, not a painted crease, reference texture, or fused eye surface.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
CENTERS=np.array([[31.3,12.,92.1],[-31.3,12.,92.1]])
SOCKET_RADIUS=12.55
BODY_DISTANCE=np.array([-2.,0.,.8,2.,3.5,5.,7.,10.])
BODY_RADIAL_MOTION=np.array([0.,0.,.25,.85,.45,-.32,-.1,0.])
BODY_CURVE=PchipInterpolator(BODY_DISTANCE,BODY_RADIAL_MOTION,extrapolate=False)

def smooth(t):
 t=np.clip(t,0.,1.);return t*t*t*(10+t*(-15+6*t))

def upper_window_y(x):
 # Same nominal original free-window design, used only as a tissue coordinate.
 h=11.6;hu=5.9;hl=3.6;rounding=.16
 width=h/np.sqrt(1-2*rounding*np.log(2)/(hu+hl));k=1-(x/width)**2
 arc=rounding*np.arccosh(np.maximum(1.,.5*np.exp(np.clip((hu+hl)*k/(2*rounding),-40,40))))
 return .065*x+.5*(hu-hl)*k+arc

def displacement(points):
 p=np.asarray(points,float);out=np.zeros_like(p)
 for sign,c in zip((1.,-1.),CENTERS):
  rel=p-c;r=np.linalg.norm(rel,axis=1);x=sign*rel[:,0];y=rel[:,1];z=rel[:,2]
  gate=smooth((27-r)/8)*smooth((z-1)/5)*smooth((14.5-np.abs(x))/3)
  # A gentle medial peak and flatter lateral descent replace the perfectly
  # even upper parabola. Spherical bearing motion preserves radial clearance.
  u=np.clip(x/11.6,-1,1);shape=(1-u*u)*(-1.5*u)
  theta=shape/10.5*smooth((y+1)/4)*gate
  ct=np.cos(theta);st=np.sin(theta)
  rotated=rel.copy();rotated[:,1]=y*ct+z*st;rotated[:,2]=-y*st+z*ct
  s=rotated[:,1]-upper_window_y(x)
  meridian=BODY_CURVE(np.clip(s,BODY_DISTANCE[0],BODY_DISTANCE[-1]))
  lateral=smooth((14-np.abs(x))/4)
  # Keep the complete R12.55 inner socket exact in radius. Tissue body begins
  # outside that inner wall and blends back into the orbital plate.
  radial_gate=smooth((r-12.8)/1.6)*gate*lateral
  fold_gate=smooth((r-14.)/2.)
  dr=np.maximum(meridian,0)*radial_gate+np.minimum(meridian,0)*radial_gate*fold_gate
  transformed=rotated*(1+dr/np.maximum(r,1e-12))[:,None]
  out+=transformed-rel
 return out

def deform(vertices,triangles,object_ids):
 v=np.asarray(vertices,float);skin=np.unique(triangles[object_ids==0]);out=v.copy()
 out[skin]+=displacement(v[skin]);return out
