"""Attachment-tangent fit inside the fresh nasal control domain.

Only the interior coefficients of the new 17x17 net are solved. Its four
boundary control curves and exact whole-domain attachment residual stay
unchanged. Smooth derivative guidance is fitted through this control domain,
with soft preservation of the new tip/alar body, not a narrow extra collar.
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from nasal_fresh_basal_control_net import (
    B,N,G,CONTROL,BOUNDARY,CONTRACT,EDGES,actual_edge,evaluate as solid_evaluate,
    OUT as SOLID_OUT,
)

OUT=SOLID_OUT/'tangent-and-finite-inlets-v2'
FREE=np.array([(i,j) for i in range(1,N-1) for j in range(1,N-1)])
_COEF=None


def basis(uv,axis=None):
    uv=np.asarray(uv,float)
    a=B(uv[:,1],1 if axis==1 else 0);b=B(uv[:,0],1 if axis==0 else 0)
    return a[:,FREE[:,0]]*b[:,FREE[:,1]]


def derivatives(uv,fn=solid_evaluate):
    uv=np.asarray(uv,float);out=[];h=1e-5
    for axis in [0,1]:
        lo=uv.copy();hi=uv.copy();lo[:,axis]=np.maximum(0,uv[:,axis]-h);hi[:,axis]=np.minimum(1,uv[:,axis]+h)
        out.append((fn(hi)-fn(lo))/(hi[:,axis]-lo[:,axis])[:,None])
    return out


def coefficients():
    global _COEF
    if _COEF is None:
        with np.load(OUT/'attachment_control_fit.npz') as q:_COEF=q['coefficients']
    return _COEF


def evaluate(uv):
    uv=np.asarray(uv,float);p=solid_evaluate(uv);p[:,2]+=basis(uv)@coefficients()
    return p


def build_fit():
    with np.load(CONTRACT) as q:original_normal=q['boundary_normals']
    a=[];rhs=[];samples=[];targets=[];source_targets=[]
    for name in EDGES:
        t=np.linspace(.003,.997,160)
        if name in ('bottom','top'):
            uv=np.c_[t,np.full(len(t),0. if name=='bottom' else 1.)];cross_axis=1;along_axis=0
        else:
            uv=np.c_[np.full(len(t),0. if name=='left' else 1.),t];cross_axis=0;along_axis=1
        edge=actual_edge(name,t)
        # The target field is sampled from actual head normals, then gently
        # regularized at about half a design unit along the attachment.
        nearest=np.argmin(np.linalg.norm(edge[:,None,:]-BOUNDARY[None,:,:],axis=2),axis=1)
        raw=original_normal[nearest];normal=gaussian_filter1d(raw,1.3,axis=0,mode='nearest')
        su,sv=derivatives(uv);jet=[su,sv];along=jet[along_axis];along/=np.linalg.norm(along,axis=1)[:,None]
        normal-=np.sum(normal*along,axis=1)[:,None]*along
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        cross=jet[cross_axis];metric=np.linalg.norm(cross[:,:2],axis=1)
        a.append(normal[:,2,None]*basis(uv,cross_axis)/metric[:,None]);rhs.append(-np.sum(normal*cross,axis=1)/metric)
        samples.append(uv);targets.append(normal);source_targets.append(raw)
    a=np.concatenate(a);rhs=np.concatenate(rhs)
    # Curvature regularization spreads boundary adjustments through the
    # whole small editable net. Central body controls remain a soft guide.
    lookup={tuple(ij):k for k,ij in enumerate(FREE)};regular=[]
    for i,j in FREE:
        for axis in [0,1]:
            row=np.zeros(len(FREE))
            for delta,w in [(-1,1),(0,-2),(1,1)]:
                ij=(i+delta,j) if axis==0 else (i,j+delta)
                if ij in lookup:row[lookup[ij]]+=w
            regular.append(row)
    regular=np.array(regular)
    y=-40+55*G[FREE[:,0]];x=(2*G[FREE[:,1]]-1)*24
    preserve=np.where((y>-32)&(y<-17)&(abs(x)<16),1.5,.03)
    lhs=40*a.T@a+.22*regular.T@regular+np.diag(preserve)
    coef=np.linalg.solve(lhs,40*a.T@rhs)
    if not np.isfinite(coef).all():raise ValueError('Invalid attachment control fit')
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'attachment_control_fit.npz',coefficients=coef,free_indices=FREE)
    global _COEF
    _COEF=coef
    uv=np.concatenate(samples);target=np.concatenate(targets);source=np.concatenate(source_targets)
    before=derivatives(uv);after=derivatives(uv,evaluate)
    angles=[]
    for u,v in [before,after]:
        n=np.cross(u,v);n/=np.linalg.norm(n,axis=1)[:,None]
        angles.append(np.rad2deg(np.arccos(np.clip(np.sum(n*target,axis=1),-1,1))))
    projected=np.rad2deg(np.arccos(np.clip(np.sum(target*source,axis=1),-1,1)))
    grid=np.stack(np.meshgrid(np.linspace(0,1,65),np.linspace(0,1,81),indexing='ij'),-1).reshape(-1,2)
    d=basis(grid)@coef
    report={'method':'Regularized tangent fit through interior coefficients of the fresh 17x17 control net; boundary position function unchanged',
        'coefficients':len(coef),'boundary_samples':len(uv),'normal_error_p50_p95_max_before':np.percentile(angles[0],[50,95,100]).tolist(),
        'normal_error_p50_p95_max_after':np.percentile(angles[1],[50,95,100]).tolist(),
        'target_projection_and_regularization_degrees_p50_p95_max':np.percentile(projected,[50,95,100]).tolist(),
        'sampled_surface_depth_change_range':[float(d.min()),float(d.max())],
        'tip_alar_soft_control_delta_range':[float(coef[preserve>1].min()),float(coef[preserve>1].max())],
        'exact_position_contract_unchanged':True,'visual_gate_passed':None,'approved_for_integration':False}
    (OUT/'attachment_fit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return coef,report


if __name__=='__main__':
    _,r=build_fit();print(json.dumps(r,indent=2),flush=True)
