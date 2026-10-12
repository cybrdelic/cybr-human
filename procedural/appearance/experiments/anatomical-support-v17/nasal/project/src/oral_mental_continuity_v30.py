"""Coordinated labial, mental and submandibular 3D soft-tissue curves.

Lip body, finite return and surrounding muzzle share a single field. This
avoids retracting a thin detached lip ribbon or freezing a trench at its edge.
A broad mental-pad adjustment and paired jaw/upper-neck support address the
long rigid lower-face silhouette together.
"""
import numpy as np
from scipy.spatial import cKDTree
from anatomical_volume_cage import build_cage,displacement as cage_displacement
from coordinated_facial_depth_v25 import smooth
ORIGIN=np.array([-84.,-140.,-28.]);STEP=7.;SHAPE=(25,19,25)

def controls(v,f,oid,semantic):
 skin=v[np.unique(f[oid==0])];roi=skin[(skin[:,1]<-27)&(skin[:,1]>-122)];tree=cKDTree(roi);rec=[]
 def add(name,p,d,weight=3.):rec.append({'region':name,'source':np.asarray(p).tolist(),'displacement':list(d),'weight':weight})
 def front(x,y):
  q=roi[(abs(roi[:,0]-x)<1.1)&(abs(roi[:,1]-y)<1.1)&(roi[:,2]>40)]
  if not len(q):raise ValueError(('No anterior sample',x,y))
  return q[np.argmax(q[:,2])]
 def pair(name,x,y,d,weight=3.):
  for sign in ([1,-1] if x else [1]):add(name,front(sign*x,y),[sign*d[0],d[1],d[2]],weight)
 lid=semantic['vermilion_vertex_ids'];upper=semantic['vermilion_upper_half']
 for row in [0,8,16,24,30]:
  t=row/30
  for column in range(0,lid.shape[1],16):
   p=v[lid[row,column]];central=max(0,1-(abs(p[0])/28)**2);is_upper=bool(upper[column])
   depth=(-2.6-.6*np.sin(np.pi*t)**2 if is_upper else -2.1-1.*np.sin(np.pi*t)**2)
   depth=depth*(.5+.5*central)
   add('coherent_labial_body_and_return',p,[-.018*p[0]*smooth((abs(p[0])-9)/18),0,depth],4.)
 for x,y,z,d in [(0,-51,112,-2.4),(15,-51,108,-2.0),(23,-51,101,-1.3),(0,-51,100,-1.6),(16,-53,101,-1.5)]:
  for sign in ([1,-1] if x else [1]):add('finite_oral_internal_support',[sign*x,y,z],[0,0,d],3.)
 for x,y,d in [(0,-37,-.5),(9,-39,-1.3),(19,-41,-1.8),(29,-45,-1.3),(34,-50,-.8),(30,-59,-.8),(18,-65,-.7),(0,-65,-.8),(38,-46,0),(40,-55,0)]:pair('broad_muzzle_transition',x,y,(0,0,d))
 for x,scale in [(0,1),(10,.85),(20,.55),(30,.2)]:
  for y,dy,dz in [(-72,.3,-1.2),(-80,1.6,-2.5),(-87,2.0,-3.3),(-94,1.5,-3.0),(-102,.6,-1.5),(-112,0,0)]:pair('rounded_mental_pad_and_submental_flow',x,y,(0,scale*dy,scale*dz))
 # Side guides straddle the previous jaw/neck join instead of preserving
 # its artistic border as a permanent ridge.
 for p,d in [((51,-66,48),(-1.1,0,-1.4)),((53,-73,27),(-1.2,0,-.8)),((48,-80,30),(.3,0,1.2)),((43,-89,36),(.6,0,1.7)),((39,-98,45),(.5,0,1.2)),((44,-107,27),(.3,0,.6))]:
  for sign in (1,-1):
   target=np.asarray(p)*[sign,1,1];actual=roi[tree.query(target)[1]];add('broad_submandibular_soft_tissue',actual,[sign*d[0],d[1],d[2]])
 for x,y in [(0,-30),(12,-31),(25,-32),(39,-34),(45,-42),(48,-55),(42,-66),(37,-110),(28,-118),(0,-119)]:pair('wide_region_return',x,y,(0,0,0),4.)
 return rec

def build(v,f,oid,semantic):return build_cage(controls(v,f,oid,semantic),origin=ORIGIN,step=STEP,shape=SHAPE)
def displacement(points,coef):
 p=np.asarray(points);d=cage_displacement(p,coef,False,origin=ORIGIN,step=STEP,shape=SHAPE)
 gate=smooth((-p[:,1]-27)/10)*smooth((p[:,1]+128)/14)*smooth((70-abs(p[:,0]))/13)*smooth((p[:,2]+5)/24)
 return d*gate[:,None]
def deform(v,f,oid,coef):
 w=v.copy();skin=np.unique(f[oid==0]);w[skin]+=displacement(v[skin],coef);return w
