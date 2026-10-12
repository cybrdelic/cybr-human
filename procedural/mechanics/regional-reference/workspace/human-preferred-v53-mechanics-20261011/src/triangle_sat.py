"""Original local triangle SAT from the preserved HUMAN numerical validation source.
Source SHA256 a89d6dbd9b37f8b31badcea34b6445837ad844d05121224bb7c9864162fb7a88.
"""
import numpy as np
def sat_intersect(a,b,eps=1e-10):
 # Separating-axis test including in-plane axes for coplanar triangle pairs.
 ae=np.roll(a,-1,axis=1)-a;be=np.roll(b,-1,axis=1)-b;na=np.cross(ae[:,0],ae[:,1]);nb=np.cross(be[:,0],be[:,1]);alive=np.ones(len(a),bool)
 axes=[na,nb]
 axes.extend(np.cross(ae[:,i],be[:,j]) for i in range(3) for j in range(3))
 axes.extend(np.cross(na,ae[:,i]) for i in range(3))
 axes.extend(np.cross(nb,be[:,i]) for i in range(3))
 for axis in axes:
  norm=np.linalg.norm(axis,axis=1);ok=norm>1e-15;axis=axis/np.maximum(norm[:,None],1e-30)
  pa=np.einsum('nvi,ni->nv',a,axis);pb=np.einsum('nvi,ni->nv',b,axis)
  separated=(pa.max(1)<pb.min(1)-eps)|(pb.max(1)<pa.min(1)-eps)
  alive &= ~(ok&separated)
 return alive
