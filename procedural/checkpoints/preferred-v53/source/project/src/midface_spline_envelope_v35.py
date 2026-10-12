"""A smooth authored midface exterior envelope with oblique anatomical guides.

This is deliberately a front-surface chart for broad cheek/maxillary planes,
not a claim to represent the entire head or its finite eye/nose/mouth anatomy.
Those remain separate connected 3D surface constructions. Direct surface
sampling removes the earlier field's overfit lobes rather than adding another
layer of sparse displacement constraints.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline,PchipInterpolator
from coordinated_facial_depth_v25 import smooth
# Oblique guide heights rise toward the lateral zygoma.
X=np.array([16.,23.,28.,35.,43.,51.,59.,67.,73.])
RIDGE_Y=np.array([-24.,-22.,-18.,-12.,-7.,-4.,-2.,0.,2.])
S=np.array([-58.,-44.,-30.,-17.,-5.,7.,19.,31.])
# Columns are lateral coordinates. The guide is broad and low-frequency.
Z=np.array([
 [110,106,101,94,84,71,54,32,12],
 [114,110,107,102,94,80,65,43,19],
 [115,112,109,105,99,88,75,54,27],
 [116,113,110,106,101,91,80,62,35],
 [117,114,110,107,102,92,80,64,38],
 [115,112,109,107,102,93,81,64,38],
 [110,109,108,106,101,92,79,58,31],
 [105,105,106,104,100,91,76,51,25],
],float)
TARGET=RectBivariateSpline(S,X,Z,kx=3,ky=3,s=0)

def deform(v,f,ids):
 w=v.copy();skin=np.unique(f[ids==0]);p=v[skin];x=abs(p[:,0]);y=p[:,1];ridge=PchipInterpolator(X,RIDGE_Y,extrapolate=True)(np.clip(x,X[0],X[-1]));s=y-ridge
 z=TARGET.ev(np.clip(s,S[0],S[-1]),np.clip(x,X[0],X[-1]));dist=np.minimum(np.linalg.norm(p-[31.3,12,94.6],axis=1),np.linalg.norm(p-[-31.3,12,94.6],axis=1))
 gate=smooth((x-17)/9)*smooth((72-x)/9)*smooth((y+81)/16)*smooth((29-y)/17)*smooth((p[:,2]-30)/25)*smooth((dist-13.4)/7)
 # Keep the correction out of finite lip tissue; the envelope blends through
 # broad oral-side support rather than pulling the vermilion into a graph.
 oral=np.sqrt((p[:,0]/31.)**2+((y+48.)/12.)**2);gate*=smooth((oral-1)/.42)
 # The chart covers the anterior cheek, not the turning mandibular sidewall.
 # Follow an oblique shared lower return with long smooth support, avoiding
 # the earlier accidental forward pull of the lateral jaw.
 lower=PchipInterpolator([17,25,35,45,55,65,73],[-80,-80,-76,-67,-54,-31,-10])(np.clip(x,17,73))
 gate*=smooth((y-lower)/20)
 w[skin,2]+=.55*gate*(z-p[:,2]);report={'method':'Direct low-resolution cubic oblique midface exterior surface, sampled with broad shared returns','x_controls':X.tolist(),'ridge_y':RIDGE_Y.tolist(),'relative_y_controls':S.tolist(),'depth_controls':Z.tolist(),'max_vertex_motion':float(np.max(abs(.55*gate*(z-p[:,2])))),'scope':'Broad front-surface chart only; finite eye, nasal and lip anatomy remains connected 3D tissue.'}
 return w,report
