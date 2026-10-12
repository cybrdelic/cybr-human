"""Fresh original nasal exterior: one small smooth 3D control net.

No old nasal interior, relaxed guide, displacement cage, or inlet plane is
imported. The only mesh input is the real shared outer attachment curve. A
17 by 17 positive-basis cubic surface carries new dorsum, paired-tip, posterior
alar-body and columellar-return controls. Its inferior chart turns in 3D.
A small, whole-domain Coons residual reproduces the actual outer curve, rather
than hiding an incompatible anatomical guide inside a narrow collar.

This is the solid exterior form gate. No openings or dense insertion yet.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BSpline,PchipInterpolator

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'output/nasal-v8-shared-insertion-study/exact_v8_cut_contract.npz'
OUT=ROOT/'output/nasal-fresh-basal-control-net-study'
N=17;DEGREE=3
K=np.r_[np.zeros(3),np.linspace(0,1,N-2),np.ones(3)]
B=BSpline(K,np.eye(N),3)
G=np.array([np.mean(K[i+1:i+4]) for i in range(N)])
U=2*G-1;Y=-40+55*G


def width(y):return 24.-12.*np.clip((np.asarray(y)+10.)/25.,0,1)


with np.load(CONTRACT) as q:
    BOUNDARY=q['boundary_points'].copy()
    BOUNDARY_IDS=q['boundary_global_vertex_ids'].copy()


def edge_data(which):
    p=BOUNDARY
    if which in ('bottom','top'):
        y=-40. if which=='bottom' else 15.
        q=p[abs(p[:,1]-y)<1e-7];parameter=(q[:,0]/width(y)+1)/2
    else:
        sign=1 if which=='right' else -1
        q=p[(abs(p[:,0]-sign*width(p[:,1]))<1e-7)]
        parameter=(q[:,1]+40)/55
    order=np.argsort(parameter);parameter=parameter[order];q=q[order]
    _,unique=np.unique(parameter,return_index=True)
    return parameter[unique],q[unique]


EDGES={name:edge_data(name) for name in ('bottom','top','left','right')}


def actual_edge(name,t):
    a,p=EDGES[name];t=np.asarray(t)
    return np.stack([np.interp(t,a,p[:,j]) for j in range(3)],axis=-1)


# Named authored section controls, from axis to paired tip, medial flank,
# inferior inlet support, convex ala, cheek return, outer sidewall and edge.
# They are spline CONTROL points, not interpolating vertex-height targets:
# the positive cubic basis rounds their transitions without spline overshoot.
GUIDE_Y=np.array([-40.,-34.,-30.,-27.,-24.,-21.,-17.,-11.,-4.,5.,15.])
GUIDE_Z=np.array([
 [121.1,121.3,121.,119.,117.,114.,111.,109.5,108.5],
 [121.,120.8,119.5,118.,116.5,114.5,111.5,109.5,108.8],
 [124.5,122.3,118.4,115.7,118.5,116.5,112.6,111.,110.7],
 [132.,129.,120.8,118.2,125.5,121.,115.3,112.3,112.2],
 [137.,138.5,133.,127.,129.3,124.,117.7,114.5,113.3],
 [137.3,140.,137.,131.,126.5,122.,118.1,114.8,113.6],
 [135.3,136.,134.,129.,124.7,120.,117.5,114.5,113.8],
 [133.2,133.,130.,126.,121.2,117.9,115.7,114.2,113.6],
 [128.,127.5,125.5,121.,117.8,114.6,112.7,111.5,112.],
 [122.4,122.,120.5,117.5,114.5,112.7,111.,109.5,108.],
 [118.8,118.5,117.6,115.7,113.7,111.8,110.5,109.5,108.6],
])


def make_controls():
    net=np.zeros((N,N,3))
    net[:,:,0]=width(Y)[:,None]*U[None,:]
    net[:,:,1]=Y[:,None]
    half=PchipInterpolator(GUIDE_Y,GUIDE_Z,axis=0)(Y)
    net[:,:,2]=np.c_[half[:,:0:-1],half]
    # The inferior wing rows turn down/back on their own 3D path. The narrow
    # central columellar curve retains a lower return into the upper lip.
    lift_y=np.array([-40.,-34.,-30.,-27.,-24.,-21.,-17.,15.])
    lift=np.zeros((len(lift_y),9))
    lift[2]=[0.,0.,.6,1.9,1.4,.35,0.,0.,0.]
    lift[3]=[-.25,0.,1.0,1.9,1.2,.15,0.,0.,0.]
    lift[4]=[0.,0.,.25,.65,.35,0.,0.,0.,0.]
    yy=PchipInterpolator(lift_y,lift,axis=0)(Y)
    net[:,:,1]+=np.c_[yy[:,:0:-1],yy]
    # Fit only the four outer control curves to the real head attachment.
    # No interior point or tangent is inherited from an earlier nasal study.
    second=np.diff(np.eye(N),n=2,axis=0)
    for name in EDGES:
        t,p=EDGES[name];a=B(t)
        c=np.linalg.solve(a.T@a+.025*second.T@second+np.eye(N)*1e-10,a.T@p)
        if name=='bottom':net[0]=c
        elif name=='top':net[-1]=c
        elif name=='left':net[:,0]=c
        else:net[:,-1]=c
    for row,v in [(0,0.),(-1,1.)]:
        for col,u in [(0,0.),(-1,1.)]:net[row,col]=actual_edge('bottom' if v==0 else 'top',np.array([u]))[0]
    return net


CONTROL=make_controls()


def raw_surface(uv):
    uv=np.asarray(uv,float)
    return np.einsum('ni,ijc,nj->nc',B(uv[:,1]),CONTROL,B(uv[:,0]),optimize=True)


def edge_residual(name,t):
    t=np.asarray(t)
    if name=='bottom':uv=np.c_[t,np.zeros_like(t)]
    elif name=='top':uv=np.c_[t,np.ones_like(t)]
    elif name=='left':uv=np.c_[np.zeros_like(t),t]
    else:uv=np.c_[np.ones_like(t),t]
    return actual_edge(name,t)-raw_surface(uv)


def evaluate(uv):
    uv=np.asarray(uv,float);u=uv[:,0,None];v=uv[:,1,None]
    # The corner residual is exactly zero because corner controls are exact.
    # Coons value correction is spread over the WHOLE chart, never a collar.
    residual=(1-v)*edge_residual('bottom',uv[:,0])+v*edge_residual('top',uv[:,0])
    residual+=(1-u)*edge_residual('left',uv[:,1])+u*edge_residual('right',uv[:,1])
    return raw_surface(uv)+residual


def build_preview():
    uv=np.stack(np.meshgrid(np.linspace(0,1,65),np.linspace(0,1,81),indexing='ij'),-1).reshape(-1,2)
    v=evaluate(uv);ind=np.arange(len(v)).reshape(65,81)
    a=ind[:-1,:-1].ravel();b=ind[1:,:-1].ravel();c=ind[1:,1:].ravel();d=ind[:-1,1:].ravel()
    f=np.r_[np.c_[a,b,c],np.c_[a,c,d]]
    tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(normal,axis=1)*.5
    if area.min()<1e-12:raise ValueError('Degenerate coarse guide')
    boundary_uv=np.c_[(BOUNDARY[:,0]/width(BOUNDARY[:,1])+1)/2,(BOUNDARY[:,1]+40)/55]
    error=float(np.max(np.linalg.norm(evaluate(boundary_uv)-BOUNDARY,axis=1)))
    if error>1e-9:raise ValueError('New guide lost actual attachment position function')
    adjustment=evaluate(uv)-raw_surface(uv)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'solid_exterior.npz',vertices=v,triangles=f,uv=uv,control_net=CONTROL,actual_boundary=BOUNDARY,actual_boundary_ids=BOUNDARY_IDS)
    report={'method':'Fresh positive-basis cubic 3D nasal control net; original named tip/alar/columellar curves; only actual outer attachment reused',
        'control_shape':list(CONTROL.shape),'vertices':len(v),'triangles':len(f),'inlets_present':False,'old_interior_mesh_or_cage_reused':False,
        'actual_outer_boundary_evaluator_maximum_error':error,'whole_domain_attachment_residual_maximum':float(np.linalg.norm(adjustment,axis=1).max()),
        'minimum_triangle_area':float(area.min()),'exterior_faces_with_posterior_normal':int(np.count_nonzero(normal[:,2]<0)),
        'visual_gate_passed':None,'approved_for_integration':False,'approved_for_selection':False,
        'scope':'Coarse solid exterior/control-net form gate only. Boundary position function is exact; tangent fit, finite inlet construction and dense topology insertion are pending.'}
    (OUT/'solid_exterior_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return v,f,report


if __name__=='__main__':
    *_,r=build_preview();print(json.dumps(r,indent=2),flush=True)
