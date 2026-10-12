"""Original coordinated globe/tissue scale control with a monotone radial return.

The independent globes, complete finite lids and sockets are uniformly scaled
inside the inner radius, then the same volumetric map returns smoothly to the
unchanged face. Support balls are disjoint. This exposes orbital proportion as
an editable anatomical authoring control, not a claim of calibrated eye size.
"""
import numpy as np
CENTERS=np.array([[-31.3,12.,94.6],[31.3,12.,94.6]])
RADIUS=12.3
INNER=17.5
OUTER=30.

def smooth(t):return t*t*t*(10-15*t+6*t*t)

def deform(v,f,ids,scale=1.07):
 if not 1<=scale<=1.15:raise ValueError('This bounded study admits scale 1 through 1.15')
 if 2*OUTER>=np.linalg.norm(CENTERS[0]-CENTERS[1]):raise ValueError('Orbital supports overlap')
 lower_bound=1-(scale-1)*1.875*OUTER/(OUTER-INNER)
 if lower_bound<=0:raise ValueError('Radial derivative conservative lower bound failed')
 w=v.copy()
 for c in CENTERS:
  d=v-c;r=np.linalg.norm(d,axis=1);mask=r<OUTER;t=np.clip((r[mask]-INNER)/(OUTER-INNER),0,1);a=1+(scale-1)*(1-smooth(t));w[mask]=c+d[mask]*a[:,None]
 globe_reports=[]
 for obj,c in zip((1,2),CENTERS):
  k=np.unique(f[ids==obj]);expected=c+(v[k]-c)*scale
  if not np.array_equal(w[k],expected):raise ValueError('Independent globe is not uniformly scaled with its interface')
  globe_reports.append({'object_id':obj,'center':c.tolist(),'radius':RADIUS*scale,'max_radial_residual':float(abs(np.linalg.norm(w[k]-c,axis=1)-RADIUS*scale).max())})
 return w,{'scale':scale,'support_inner':INNER,'support_outer':OUTER,'conservative_continuous_radial_derivative_lower_bound':lower_bound,'support_balls_disjoint':True,'globes':globe_reports,'scope':'Original reversible proportion study; continuous map is monotone and disjoint, while discrete mesh and optical interfaces still require their own checks. No calibrated anatomical measurements.'}
