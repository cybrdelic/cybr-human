"""Shared oral support and finite-section proportion control.

One smooth volume map carries all lip, oral cavity and local host vertices.
It does not separately translate an isolated vermilion island.
"""
import numpy as np

def smooth(t):
 t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))

def displacement(p):
 x,y,z=p.T;ax=np.abs(x)
 # Existing authored oral seam, retained across full width.
 seam=-51.10-.52*(ax/27)**2
 dy=y-seam
 xgate=smooth((46-ax)/14)
 front=smooth((z-62)/24)
 # Translate the shared oral base down 3.5 units; long blending ramps
 # preserve the smooth philtral and mental attachments.
 nasal_release=smooth((-35-y)/10)
 carrier=smooth((y+103)/24)*nasal_release*xgate*front
 translate=-3.5*carrier
 # Compress upper vermilion by 30%, lower by 10%, with smooth centerline
 # transition and broad release outside the finite oral section.
 section=smooth((22-np.abs(dy))/10)*nasal_release*xgate*front
 compression=(.20+.10*np.tanh(dy/1.5))*dy
 d=np.zeros_like(p);d[:,1]=translate-section*compression
 return d

def reshape_oral_support(p):
 d=displacement(p);q=p+d
 # Dense regular samples check the only changing diagonal entry dy'/dy.
 grid=np.stack(np.meshgrid(np.linspace(-48,48,41),np.linspace(-105,-22,121),np.linspace(62,146,25),indexing='ij'),-1).reshape(-1,3)
 h=np.array([0,.001,0]);derivative=1+(displacement(grid+h)[:,1]-displacement(grid-h)[:,1])/.002
 assert derivative.min()>.45
 return q,{'construction':'Shared C2-volume oral support translation and smoothly varying upper/lower finite-section height controls','inferior_translation':3.5,'upper_height_scale':.70,'lower_height_scale':.90,'mouth_width_unchanged':True,'sampled_minimum_y_jacobian':float(derivative.min()),'sampled_maximum_y_jacobian':float(derivative.max()),'maximum_motion':float(np.linalg.norm(d,axis=1).max()),'finite_oral_and_host_share_map':True,'nasal_base_protected_above_y':-35.0,'full_oral_support_below_y':-45.0}
