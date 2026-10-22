"""Small local triangle queries for anatomical attachment construction.

Centroid candidate acceleration is approximate, not a global contact guarantee.
"""
import numpy as np
from scipy.spatial import cKDTree

class SurfaceIndex:
 """Owned static triangles/centroid tree; rebuild after any body deformation."""
 def __init__(self,vertices,faces):
  self.triangles=np.asarray(vertices)[np.asarray(faces)].copy()
  self.tree=cKDTree(self.triangles.mean(1))

def closest_surface(points,vertices,faces,candidates=48,surface_index=None):
 p=np.atleast_2d(points)
 if len(p)>4096:
  index=SurfaceIndex(vertices,faces) if surface_index is None else surface_index
  rows=[closest_surface(p[i:i+4096],vertices,faces,candidates,index) for i in range(0,len(p),4096)]
  return tuple(np.concatenate([r[j] for r in rows]) for j in range(4))
 tri=np.asarray(vertices)[np.asarray(faces)] if surface_index is None else surface_index.triangles
 tree=cKDTree(tri.mean(1)) if surface_index is None else surface_index.tree
 k=min(candidates,len(tri));idx=tree.query(p,k=k)[1]
 if k==1:idx=idx[:,None]
 a,b,c=(tri[idx][:,:,i] for i in range(3));pp=p[:,None,:];ab=b-a;ac=c-a;ap=pp-a
 d00=(ab*ab).sum(-1);d01=(ab*ac).sum(-1);d11=(ac*ac).sum(-1);d20=(ap*ab).sum(-1);d21=(ap*ac).sum(-1);den=d00*d11-d01*d01
 u=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=abs(den)>1e-20);v=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=abs(den)>1e-20)
 bary=np.stack([1-u-v,u,v],-1);q=a+u[:,:,None]*ab+v[:,:,None]*ac;valid=(bary.min(-1)>=0)&(abs(den)>1e-20)
 dist=((q-pp)**2).sum(-1);dist[~valid]=np.inf
 for ia,ib in [(0,1),(1,2),(2,0)]:
  aa=tri[idx][:,:,ia];bb=tri[idx][:,:,ib];edge=bb-aa;t=np.clip(((pp-aa)*edge).sum(-1)/np.maximum((edge*edge).sum(-1),1e-20),0,1);qq=aa+t[:,:,None]*edge;dd=((qq-pp)**2).sum(-1);take=dd<dist
  bar=np.zeros_like(bary);bar[:,:,ia]=1-t;bar[:,:,ib]=t
  q[take]=qq[take];bary[take]=bar[take];dist[take]=dd[take]
 best=np.argmin(dist,1);r=np.arange(len(p));return q[r,best],idx[r,best],bary[r,best],np.sqrt(dist[r,best])

def winding_inside(points,vertices,faces,chunk=192):
 """Floating-point generalized winding for a small closed anatomical surface."""
 tri=np.asarray(vertices)[np.asarray(faces)];result=[]
 for start in range(0,len(points),chunk):
  p=np.asarray(points)[start:start+chunk];a,b,c=(tri[None,:,:, :]-p[:,None,None,:]).transpose(2,0,1,3)
  la=np.linalg.norm(a,axis=2);lb=np.linalg.norm(b,axis=2);lc=np.linalg.norm(c,axis=2)
  num=np.einsum('...i,...i->...',a,np.cross(b,c));den=la*lb*lc+np.einsum('...i,...i->...',a,b)*lc+np.einsum('...i,...i->...',b,c)*la+np.einsum('...i,...i->...',c,a)*lb
  result.extend(abs(np.arctan2(num,den).sum(1))>np.pi)
 return np.asarray(result,bool)
