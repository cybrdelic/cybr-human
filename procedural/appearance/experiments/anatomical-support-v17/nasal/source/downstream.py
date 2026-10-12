from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT/'project';sys.path.insert(0,str(PROJECT/'src'))
import coordinated_facial_depth_v25 as depth
import midface_cross_section_planes_v28 as transverse
import midface_spline_envelope_v35 as envelope
import oral_mental_continuity_v30 as oral
import upper_lid_rest_pose_v36 as rest
import coherent_orbital_scale_v53 as scale
_COEFFICIENTS=None
def downstream(p):
 global _COEFFICIENTS
 if _COEFFICIENTS is None:
  C=PROJECT/'output/coordinated-cervical-v31/controls';_COEFFICIENTS={name:np.load(C/(name+'_cage.npz'))['coefficients'] for name in ['cheek','support','midface','oral']}
 c=_COEFFICIENTS;w=p.copy();w+=depth.cheek_displacement(w,c['cheek']);w+=depth.SUPPORT_FRACTION*depth.support_displacement(w,c['support']);w+=depth.orbital_displacement(w);w+=.68*transverse.displacement(w,c['midface']);w+=oral.displacement(w,c['oral']);w,_=envelope.deform(w,np.arange(len(w))[:,None],np.zeros(len(w),np.uint8));w=rest.move_points(w)
 for center in scale.CENTERS:
  d=w-center;r=np.linalg.norm(d,axis=1);mask=r<scale.OUTER;t=np.clip((r[mask]-scale.INNER)/(scale.OUTER-scale.INNER),0,1);w[mask]=center+d[mask]*(1+.07*(1-scale.smooth(t)))[:,None]
 return w
