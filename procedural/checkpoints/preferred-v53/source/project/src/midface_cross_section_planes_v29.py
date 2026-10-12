"""Bounded first application of the coordinated transverse plane redesign.

The full v28 target change failed the conservative local orientation gate.
This candidate applies 68 percent as a coherent single step; artistic targets
stay editable and no individual vertices are clipped.
"""
import numpy as np
from midface_cross_section_planes_v28 import build,displacement
STRENGTH=.68

def deform(v,f,ids,coef):
 w=v.copy();skin=np.unique(f[ids==0]);w[skin]+=STRENGTH*displacement(v[skin],coef);return w
