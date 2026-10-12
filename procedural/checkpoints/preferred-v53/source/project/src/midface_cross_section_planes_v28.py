"""Reference-guided midface cross-sectional plane redesign.

The prior surface has an almost constant anterior depth from medial cheek
through the lateral zygoma, followed by an abrupt hollow at the mouth corner.
Explicit broad transverse depth guides replace that plateau/trough relationship.
A regularized three-dimensional cage carries the existing connected skin.
No textures or independent overlay surfaces are involved.
"""
import numpy as np
from anatomical_volume_cage import build_cage,displacement as cage_displacement
from coordinated_facial_depth_v25 import smooth
CENTERS=np.array([[31.3,12.,94.6],[-31.3,12.,94.6]])
ORIGIN=np.array([-98.,-91.,-14.]);STEP=7.;SHAPE=(29,23,25)
# Curves are exploratory artistic proportions, not recovered physical anatomy.
X=np.array([20,25,30,35,40,45,50,55,60,65],float)
ROWS=[
 (8,[104,106,107,106,103,97,91,83,71,55]),
 (0,[111,109,107.5,105.5,102,97,91,84,74,59]),
 (-8,[114,111,109,107,104,99,93,85,75,61]),
 (-16,[116,112,109,107,103,98,91,82,71,57]),
 (-24,[120,113,109,106,102,96,88,78,66,50]),
 (-32,[117,113,109,105,100,93,84,74,61,44]),
 (-40,[115,112,108,103,97,89,79,68,54,38]),
 (-48,[117,114,110,105,99,91,80,66,50,34]),
 (-56,[116,113,109,104,97,87,75,61,44,30]),
]

def controls(v,f,ids):
 skin=v[np.unique(f[ids==0])];skin=skin[(skin[:,2]>20)&(abs(skin[:,0])<74)&(skin[:,1]>-72)&(skin[:,1]<40)];rec=[]
 def sample(x,y):
  q=skin[(abs(skin[:,0]-x)<1)&(abs(skin[:,1]-y)<1)]
  if not len(q):return None
  return q[np.argmax(q[:,2])]
 for y,zs in ROWS:
  for x,z in zip(X,zs):
   for sign in (1,-1):
    p=sample(sign*x,y)
    if p is None:continue
    # Do not author a target behind the lateral boundary at the current
    # section; far lateral guide changes end before the ear assembly.
    rec.append({'region':'malar_buccal_transverse_plane','source':p.tolist(),'displacement':[0,0,float(z-p[2])],'weight':3.})
 for x,y in [(15,8),(15,0),(15,-12),(16,-22),(16,-33),(16,-42),(16,-53),(20,-64),(30,-68),(40,-68),(52,-62),(61,-48),(68,-32),(69,-8),(66,14),(59,24),(49,29),(37,29),(25,29)]:
  for sign in (1,-1):
   p=sample(sign*x,y)
   if p is not None:rec.append({'region':'broad_region_return','source':p.tolist(),'displacement':[0,0,0],'weight':3.})
 return rec

def build(v,f,ids):return build_cage(controls(v,f,ids),origin=ORIGIN,step=STEP,shape=SHAPE)
def displacement(points,coef):
 p=np.asarray(points);d=cage_displacement(p,coef,False,origin=ORIGIN,step=STEP,shape=SHAPE)
 dist=np.minimum(np.linalg.norm(p-CENTERS[0],axis=1),np.linalg.norm(p-CENTERS[1],axis=1))
 gate=smooth((dist-13.4)/6.5)*smooth((abs(p[:,0])-15)/6)*smooth((75-abs(p[:,0]))/9)*smooth((p[:,1]+73)/12)*smooth((35-p[:,1])/12)*smooth((p[:,2]-20)/18)
 return d*gate[:,None]
def deform(v,f,ids,coef):
 w=v.copy();skin=np.unique(f[ids==0]);w[skin]+=displacement(v[skin],coef);return w
