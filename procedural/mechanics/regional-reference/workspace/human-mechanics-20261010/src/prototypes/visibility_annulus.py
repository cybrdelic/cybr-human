"""One fixed-point visibility-constrained annulus triangulation."""
import numpy as np
def visibility_zipper(bot,top,x,center):
 n,m=len(bot),len(top);br=np.array([[[bot[i],bot[(i+1)%n],top[j]] for j in range(m)] for i in range(n)]);bt=np.array([[[bot[i],top[(j+1)%m],top[j]] for j in range(m)] for i in range(n)]);qs=[]
 for faces in [br,bt]:
  z=x[faces];normal=np.cross(z[:,:,1]-z[:,:,0],z[:,:,2]-z[:,:,0]);height=np.einsum('ijk,ijk->ij',normal,z[:,:,0]-center)/np.linalg.norm(normal,axis=2);qs.append(height)
 best=None
 for si in range(n):
  for sj in range(m):
   score=np.full((n+1,m+1),-np.inf);score[0,0]=100;prev=np.zeros((n+1,m+1),np.int8)
   for i in range(n+1):
    for j in range(m+1):
     if score[i,j]<=1e-5:continue
     if i<n:
      q=min(score[i,j],qs[0][(i+si)%n,(j+sj)%m])
      if q>max(1e-5,score[i+1,j]):score[i+1,j]=q;prev[i+1,j]=1
     if j<m:
      q=min(score[i,j],qs[1][(i+si)%n,(j+sj)%m])
      if q>max(1e-5,score[i,j+1]):score[i,j+1]=q;prev[i,j+1]=2
   if score[n,m]<=1e-5 or (best and score[n,m]<=best[0]):continue
   i,j=n,m;faces=[]
   while i or j:
    if prev[i,j]==1:i-=1;faces.append(br[(i+si)%n,(j+sj)%m])
    elif prev[i,j]==2:j-=1;faces.append(bt[(i+si)%n,(j+sj)%m])
    else:raise RuntimeError('Bad path')
   best=(float(score[n,m]),np.array(faces[::-1]),si,sj)
 return best
