"""Copied original bounded triangle checks, unchanged algorithms."""
import numpy as np

def nearest_points_batch(p,tri):
 A,B,C=tri[:,0],tri[:,1],tri[:,2];AB=B-A;AC=C-A;AP=p-A
 n=np.cross(AB,AC);nn=np.einsum('ij,ij->i',n,n);proj=p-n*(np.einsum('ij,ij->i',AP,n)/nn)[:,None]
 V=proj-A;d00=np.einsum('ij,ij->i',AB,AB);d01=np.einsum('ij,ij->i',AB,AC);d11=np.einsum('ij,ij->i',AC,AC);d20=np.einsum('ij,ij->i',V,AB);d21=np.einsum('ij,ij->i',V,AC)
 den=d00*d11-d01*d01;u=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
 d=np.linalg.norm(proj-p,axis=1);d[(u<0)|(w<0)|(u+w>1)]=np.inf
 for x,y in ((A,B),(B,C),(C,A)):
  edge=y-x;t=np.clip(np.einsum('ij,ij->i',p-x,edge)/np.einsum('ij,ij->i',edge,edge),0,1);q=x+t[:,None]*edge;dd=np.linalg.norm(q-p,axis=1);take=dd<d;d[take]=dd[take];proj[take]=q[take]
 return d,proj

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
