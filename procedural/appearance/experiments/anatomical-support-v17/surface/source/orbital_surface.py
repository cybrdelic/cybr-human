"""Fresh full-width XYZ ocular sections on exact original host loops."""
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
R=13.161

def normalize(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-15)
def periodic(x,y):
    return CubicSpline(np.r_[x,x[0]+2*np.pi],np.r_[y,y[:1]],axis=0,bc_type='periodic')
def connect(a,pa,b,pb):
    """Positive-XY outer-to-inner zipper, preserving arbitrary exact boundary count."""
    # Both loops have monotonically increasing phase in [0,2pi).
    i=j=0;faces=[];na=len(a);nb=len(b)
    while i<na or j<nb:
        xa=pa[(i+1)%na]+(2*np.pi if i+1>=na else 0) if i<na else np.inf
        xb=pb[(j+1)%nb]+(2*np.pi if j+1>=nb else 0) if j<nb else np.inf
        if xa<xb:
            faces.append([a[i%na],a[(i+1)%na],b[j%nb]]);i+=1
        else:
            faces.append([a[i%na],b[(j+1)%nb],b[j%nb]]);j+=1
    return np.array(faces,np.int64)
def build(side,sign):
    cut=np.load(OUT/'interfaces'/f'{side}_cut.npz');boundary=cut['boundary_points'];normal=cut['boundary_normals'];c=np.array([sign*31.3,12,94.6])
    d=boundary-c;phase=np.mod(np.arctan2(d[:,1],sign*d[:,0]),2*np.pi);order=np.argsort(phase);phase=phase[order];boundary=boundary[order];normal=normal[order]
    old_edges={tuple(sorted((int(a),int(b)))) for a,b in zip(order,np.roll(order,-1))};expected={tuple(sorted((i,(i+1)%len(order)))) for i in range(len(order))};assert old_edges==expected
    theta=np.mod(2*np.pi*np.arange(256)/256+phase[0],2*np.pi);theta.sort()
    # Exact positional host interpolation. Angularly regularized tangent guides
    # are confined to the outer continuation, never carried into lid contours.
    tp=np.where(theta<phase[0],theta+2*np.pi,theta)
    h=np.column_stack([np.interp(tp,np.r_[phase,phase[0]+2*np.pi],np.r_[boundary[:,j],boundary[0,j]]) for j in range(3)])
    nn=np.column_stack([np.interp(tp,np.r_[phase,phase[0]+2*np.pi],np.r_[normal[:,j],normal[0,j]]) for j in range(3)])
    n=normalize(gaussian_filter1d(nn,6,axis=0,mode='wrap'))
    xx=np.where(np.cos(theta)<0,12.65,12.70)*np.cos(theta)
    sine=np.sin(theta);arc=sine*(.30+.70*abs(sine)**.60)
    yy=.065*xx+np.where(sine>=0,5.80*(1-.14*np.cos(theta)),4.35)*arc
    wet_extra=.25*np.exp(-(np.sqrt(xx*xx+yy*yy)/7.5)**4)
    wet=c+np.c_[sign*xx,yy,np.sqrt((R+.28+wet_extra)**2-xx*xx-yy*yy)]
    nasal_corner=np.exp(-((theta-np.pi)/.46)**2)
    wet[:,2]+=2.60*nasal_corner
    wet_radius=np.linalg.norm(wet-c,axis=1)
    ex=xx*1.014;ey=.065*ex+(yy-.065*xx)*1.045
    central_z=np.sqrt((R+.62)**2-ex*ex-ey*ey)
    terminal=np.exp(-(np.sin(theta)/.23)**2)
    tx=np.where(np.cos(theta)<0,-.55,.20)*terminal
    tz=np.where(np.cos(theta)<0,5.20,5.00)
    rim=c+np.c_[sign*(ex+tx),ey,(1-terminal)*central_z+terminal*tz+2.60*nasal_corner]
    delta=h-rim;distance=np.linalg.norm(delta[:,:2],axis=1)
    upper=np.maximum(sine,0)**2;lower=np.maximum(-sine,0)**2
    # Independent common contours: margin, tarsal/lower crest, shallow upper
    # crease/lower continuation, brow/cheek bearing, exact outer host.
    fractions=np.stack([np.zeros(256),np.minimum(2.0/distance,.27),np.minimum(4.3/distance,.53),np.minimum(8.0/distance,.78),np.ones(256)])
    controls=rim[None]+fractions[:,:,None]*delta[None]
    base1=controls[1,:,2].copy();base2=controls[2,:,2].copy()
    controls[1,:,2]=base1*(1-upper-lower)+upper*(rim[:,2]-.35)+lower*(base1+.18)
    controls[2,:,2]=base2*(1-upper-lower)+upper*(rim[:,2]-1.45)+lower*(base2+.14)
    # The upper crease ends before the supported corners. No circumferential
    # crease ring is created. Distal brow/cheek are independently controlled.
    controls[3,:,2]=.55*controls[3,:,2]+.45*((1-fractions[3])*controls[2,:,2]+fractions[3]*h[:,2])
    # Coupled periodic contour fairness; the exact rim and outer boundary stay fixed.
    controls[1:4,:,2]=gaussian_filter1d(controls[1:4,:,2],1.4,axis=1,mode='wrap')
    knots=np.array([0.,.18,.37,.62,1.])
    slopes=PchipInterpolator(knots,controls,axis=0).derivative()(knots)
    # Explicit upper crest and crease tangents distinguish body, fold and brow.
    slopes[1,:,2]*=(1-.15*upper)
    slopes[2,:,2]*=(1-upper)
    # Local terminal wedge slopes die at the first body contour; no correction
    # fans out to the host. Host depth tangent enters only the last chart band.
    hnz=np.maximum(n[:,2],.40)
    outer=slopes[-1,:,:2]
    slopes[-1,:,2]=-(n[:,0]*outer[:,0]+n[:,1]*outer[:,1])/hnz
    slopes[-1,:,2]=gaussian_filter1d(slopes[-1,:,2],4,mode='wrap')
    # Local anterior crest; outer body/crease/brow controls and partials remain exact.
    old_rim=rim.copy();old_d0=slopes[0].copy()
    rim=rim+np.c_[np.zeros(256),.10*upper-.16*lower,.20*upper+.44*lower]
    controls[0]=rim
    slopes[0,:,2]=(1-upper-lower)*old_d0[:,2]+upper*(.62*old_d0[:,2])+lower*(-1.6)
    body=CubicHermiteSpline(knots,controls,slopes,axis=0)
    d0=slopes[0]
    samples=np.unique(np.r_[np.linspace(0,1,37),knots,.045,.10])[::-1]
    body_points=body(samples)
    blocks=[boundary];faces=[];curves={'host_boundary':np.arange(len(boundary))};previous=curves['host_boundary'];pp=phase;cursor=len(boundary);parts=[]
    for t,p in zip(samples[1:],body_points[1:]):
        ids=np.arange(cursor,cursor+len(theta));cursor+=len(theta);blocks.append(p);ff=connect(previous,pp,ids,theta);faces.append(ff);parts.append(np.zeros(len(ff),np.uint8));previous=ids;pp=theta
        if abs(t-.045)<1e-8:curves['lash_support']=ids
        if abs(t-.18)<1e-8:curves['body_bearing']=ids
        if abs(t-.37)<1e-8:curves['upper_crease_guide']=ids
        if abs(t-.62)<1e-8:curves['brow_cheek_guide']=ids
    curves['external_margin']=previous;curves['plate_inner_boundary']=curves['body_bearing']
    # Finite U-turn from the visible rim to the wetline. End tangent points
    # outward along the globe into the closed socket; both endpoints are shared.
    rdir=normalize(rim-c);wdir=normalize(wet-c)
    az=np.arctan2(sign*wdir[:,0],wdir[:,1])
    rho=np.linalg.norm((wet-c)[:,:2],axis=1);beta=np.arctan2(rho,(wet-c)[:,2]);planar=normalize((wet-c)[:,:2])
    sphere_tangent=np.c_[planar*np.cos(beta)[:,None],-np.sin(beta)]
    # Two joined finite sections: short anterior arc and posterior wet return.
    # The original turn is exactly subdivided at the supported terminal wedges.
    old_a=old_rim-normalize(old_d0)*.30;old_b=wet-sphere_tangent*.22
    q0=(old_rim+old_a)/2;q1=(old_a+old_b)/2;q2=(old_b+wet)/2
    p0=(q0+q1)/2;p1=(q1+q2)/2;old_mid=(p0+p1)/2
    weight=upper+lower;inward_y=-np.sign(sine)
    lip_target=rim+np.c_[np.zeros(256),inward_y*(.22*upper+.30*lower)/np.maximum(weight,1e-15),-(.07*upper+.08*lower)/np.maximum(weight,1e-15)]
    lip=(1-weight[:,None])*old_mid+weight[:,None]*lip_target
    join_target=normalize(np.c_[np.zeros(256),inward_y,-np.ones(256)])*.08
    join=(1-weight[:,None])*(p1-old_mid)+weight[:,None]*join_target
    front_a=rim-normalize(d0)*(.15-.05*weight)[:,None]
    front_b=lip-join;back_a=lip+join;back_b=q2
    for half,(aa,bb,cc,dd) in enumerate([(rim,front_a,front_b,lip),(lip,back_a,back_b,wet)]):
        for t in np.linspace(0,1,7)[1:]:
            p=(1-t)**3*aa+3*(1-t)**2*t*bb+3*(1-t)*t*t*cc+t**3*dd
            ids=np.arange(cursor,cursor+256);cursor+=256;blocks.append(p);ff=connect(previous,theta,ids,theta);faces.append(ff);parts.append(np.ones(len(ff),np.uint8));previous=ids
        if half==0:curves['anterior_lip_turn']=previous
    curves['inner_wetline']=previous
    for t in np.linspace(0,1,41)[1:]:
        angle=beta+t*(np.pi-.065-beta)
        p=c+(R+.28+(wet_radius-(R+.28))*(1-t)**8)[:,None]*np.c_[planar*np.sin(angle)[:,None],np.cos(angle)]
        ids=np.arange(cursor,cursor+256);cursor+=256;blocks.append(p);ff=connect(previous,theta,ids,theta);faces.append(ff);parts.append(np.full(len(ff),2,np.uint8));previous=ids
    pole=cursor;blocks.append((c+[0,0,-R-.28])[None]);ff=np.c_[previous,np.roll(previous,-1),np.full(256,pole)];faces.append(ff);parts.append(np.full(256,2,np.uint8))
    v=np.concatenate(blocks);f=np.concatenate(faces);part=np.concatenate(parts)
    if sign<0:f=f[:,[0,2,1]]
    # Ordered semantic loops retain source phase and declared upper/lower labels.
    data={'vertices':v,'triangles':f,'face_parts':part,'boundary_source_order':order,'boundary_combined_vertex_ids':cut['boundary_combined_vertex_ids'][order],'globe_center':c,'globe_radius':np.array(R),'semantic_theta':theta,'semantic_upper_half':sine>=0}
    data.update({k+'_vertex_ids':ids for k,ids in curves.items()})
    data.update(contour_positions=controls, contour_parameters=knots, contour_derivatives=slopes, host_guide_normals=n, body_sample_parameters=samples, body_sample_positions=body_points, old_margin_positions=old_rim, anterior_lip_positions=lip, margin_shift_weight=weight)
    return data
