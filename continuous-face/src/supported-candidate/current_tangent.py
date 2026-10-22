"""Exact restored neo-Hookean active-strain tangent; no contact linearization."""
import numpy as np
from scipy.sparse import coo_matrix
def tangent(s,q,activation=0):
 T=s.T;F=np.stack([q[T[:,j]]-q[T[:,0]] for j in [1,2,3]],axis=2)@s.inv;J=np.linalg.det(F);it=np.linalg.inv(F).transpose(0,2,1)
 A=np.broadcast_to(np.eye(3),(len(T),3,3)).copy();gamma=1-.2*activation;f=s.fibre[s.active];A[s.active]=np.sqrt(gamma)*np.eye(3)+(1/gamma-np.sqrt(gamma))*f[:,:,None]*f[:,None,:];F=F@A;J=np.linalg.det(F);it=np.linalg.inv(F).transpose(0,2,1)
 grad=np.concatenate([-s.inv.sum(1)[:,None,:],s.inv],axis=1)@A;v=np.einsum('tij,taj->tai',it,grad);dot=np.einsum('tai,tbi->tab',grad,grad)
 H=s.volume[:,None,None,None,None]*(s.mu[:,None,None,None,None]*np.einsum('tab,ij->taibj',dot,np.eye(3))+s.lam[:,None,None,None,None]*np.einsum('tai,tbj->taibj',v,v)-(s.lam*np.log(J)-s.mu)[:,None,None,None,None]*np.einsum('tbi,taj->taibj',v,v))
 dof=(T[:,:,None]*3+np.arange(3)).reshape(-1,12);return coo_matrix((H.ravel(),(np.repeat(dof,12,axis=1).ravel(),np.tile(dof,(1,12)).ravel())),shape=(len(q)*3,)*2).tocsr()
