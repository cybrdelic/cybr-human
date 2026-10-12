"""Coordinated oral arch, nasal projection, mental pad and cranial-arc controls.

Built on the frozen v5 mesh. Source landmarks come from our geometry; target
motions are approximate image-guided artistic choices, not calibrated anatomy.
No separate imported surface, texture or render-dependent displacement is used.
"""
from pathlib import Path
import numpy as np
from anatomical_volume_cage import build_cage,displacement

R=Path(__file__).resolve().parents[1]
LOW_ORIGIN=np.array([-64.,-136.,32.]);LOW_STEP=6.;LOW_SHAPE=(23,20,24)
NOSE_ORIGIN=np.array([-48.,-68.,72.]);NOSE_STEP=6.;NOSE_SHAPE=(17,19,16)
CROWN_ORIGIN=np.array([-112.,8.,-104.]);CROWN_STEP=12.;CROWN_SHAPE=(20,13,21)

def smooth(t):
    t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))

def control_record(region,p,d,weight=2.):
    return {'region':region,'source':np.asarray(p).tolist(),'displacement':np.asarray(d).tolist(),'weight':weight}

def front_sample(v,x,y):
    # A bounded actual frontal surface sample, not the interior oral wall.
    sel=(np.abs(v[:,0]-x)<.85)&(np.abs(v[:,1]-y)<.85)&(v[:,2]>40)
    q=v[sel]
    if not len(q):raise ValueError(f'No anterior sample near {x},{y}')
    return q[np.argmax(q[:,2])]

def lower_controls(v):
    rec=[]
    # Broad mental eminence relative to the labiomental hollow. Preserve the
    # mandibular angle; the motion is concentrated on the pad and upper throat.
    for x,width in [(0,1.),(10,.85),(20,.42),(30,.08)]:
        for y,dy,dz in [(-70,0,-1.5),(-77,0,.2),(-85,-.15,.8),
                         (-93,-.45,.8),(-98,-.65,.6)]:
            for sign in ([1,-1] if x else [1]):
                p=front_sample(v,sign*x,y)
                rec.append(control_record('mental_pad_profile',p,[0,width*dy,width*dz],2.5))
    for x,width in [(0,1.),(10,.8),(20,.35)]:
        for y,dy,dz in [(-103,-1.8,1.3),(-108,-1.0,.8),(-116,0,0),(-128,0,0)]:
            for sign in ([1,-1] if x else [1]):
                p=front_sample(v,sign*x,y)
                rec.append(control_record('continuous_upper_throat',p,[0,width*dy,width*dz],2.3))
    with np.load(R/'output/coordinated-gestalt-v5/anatomical_semantics/authored_lip_lid_semantics.npz') as q:
        ids=q['vermilion_vertex_ids'];upper=q['vermilion_upper_half']
    # Narrow predominantly the lateral mouth. Add vertical central tissue
    # volume while holding the oral seam, rather than raising it once again.
    for row in [0,8,16,24,30]:
        t=row/30.
        for column in range(0,384,16):
            p=v[ids[row,column]];x=abs(p[0]);central=max(0.,1-(x/28.)**2)
            side=1. if upper[column] else -1.
            dx=-.045*p[0]*smooth((x-7)/18)
            dy=side*.8*central*(1-t)
            dz=-3.3+.5*min(x/28.,1.)**2 + (.35 if upper[column] else .55)*central*np.sin(np.pi*t)**2
            rec.append(control_record('recessed_curved_oral_arch',p,[dx,dy,dz],3.5))
    # Carry the lips into an actual oral mound, with a receding lateral arch.
    # Retraction moves finite inner tissue with the exterior; it is not a
    # surface-only displacement or a rendered shading effect.
    for x,y,dz in [(0,-38,-2.7),(10,-39,-3.0),(20,-41,-2.9),(28,-43,-2.5),
                    (32,-50,-2.3),(33,-59,-2.0),(26,-65,-2.1),(15,-67,-1.7),
                    (0,-66,-1.5),(38,-47,-1.3),(38,-61,-1.0)]:
        for sign in ([1,-1] if x else [1]):
            rec.append(control_record('oral_mound_support',front_sample(v,sign*x,y),[0,0,dz],3.))
    for x,y,dz in [(43,-43,-.5),(44,-55,-.35),(44,-67,-.25),(43,-80,0),(35,-91,0),(25,-105,0),(30,-116,0)]:
        for sign in ([1,-1] if x else [1]):
            rec.append(control_record('gradual_lateral_oral_return',front_sample(v,sign*x,y),[0,0,dz],3.))
    return rec

def nose_controls(v):
    rec=[]
    # Recede the whole tip/alar/vestibule complex. The bridge tapers into the
    # separate brow field; the lower columella blends with the oral mound.
    for x,y,dz in [(0,18,0),(0,10,-.7),(0,0,-1.6),(0,-10,-2.6),
                    (0,-20,-3.3),(0,-29,-3.3),(0,-36,-2.3),
                    (7,-5,-1.9),(10,-16,-2.8),(14,-26,-3.3),
                    (17,-32,-2.8),(9,-35,-2.5),(22,-28,-1.2),
                    (28,-27,0),(24,-12,0),(18,8,0),(0,-44,0)]:
        for sign in ([1,-1] if x else [1]):
            p=front_sample(v,sign*x,y)
            rec.append(control_record('nasal_projection_framework',p,[0,0,dz],3.))
    # Internal tissue follows the same broad anterior-to-posterior motion,
    # rather than leaving a fixed vestibule behind a moving nasal entrance.
    for x,y,z,dz in [(0,-24,120,-2.7),(10,-29,121,-3.0),(13,-32,116,-2.3),
                     (0,-15,112,-1.8),(0,-35,110,-1.6),(24,-28,105,0)]:
        for sign in ([1,-1] if x else [1]):
            rec.append(control_record('nasal_internal_support',[sign*x,y,z],[0,0,dz],2.))
    return rec

def crown_controls(v):
    rec=[]
    # Restore a broad frontal/parietal arc, not a single raised crown point.
    # The face, eyes and chin remain fixed. This partially reverses the earlier
    # exploratory crown shrink and improves the normalized feature placement.
    for x,y,z,dy in [(0,104,16,4.5),(28,98,20,4.0),(48,85,22,3.0),
                      (0,87,80,3.0),(30,84,75,2.9),(49,75,62,2.2),
                      (0,81,-47,3.0),(30,75,-43,2.6),(48,62,-32,1.8),
                      (0,62,107,1.0),(35,57,90,.8),(55,45,65,.3)]:
        for sign in ([1,-1] if x else [1]):
            rec.append(control_record('frontal_parietal_arc',[sign*x,y,z],[0,dy,0],2.5))
    for x,y,z in [(0,35,119),(25,35,117),(50,32,92),(65,20,47),
                   (0,35,-61),(35,30,-54),(65,15,-3)]:
        for sign in ([1,-1] if x else [1]):
            rec.append(control_record('retained_lower_cranium',[sign*x,y,z],[0,0,0],4.))
    return rec

def evaluate_displacement(points,coefficients):
    q=np.asarray(points,float)
    low=displacement(q,coefficients['lower'],origin=LOW_ORIGIN,step=LOW_STEP,shape=LOW_SHAPE,eye_center_z=92.1)
    top=displacement(q,coefficients['crown'],origin=CROWN_ORIGIN,step=CROWN_STEP,shape=CROWN_SHAPE,eye_center_z=92.1)
    top*=smooth((q[:,1]-35)/30)[:,None]
    nose=displacement(q,coefficients['nose'],origin=NOSE_ORIGIN,step=NOSE_STEP,shape=NOSE_SHAPE,eye_center_z=92.1)
    return low+top+nose

def deform(vertices,triangles,object_ids):
    v=np.asarray(vertices,float);f=np.asarray(triangles);ids=np.asarray(object_ids)
    skin=np.unique(f[ids==0]);eye=np.unique(f[ids>0]);q=v[skin]
    lower,lr=build_cage(lower_controls(v),origin=LOW_ORIGIN,step=LOW_STEP,shape=LOW_SHAPE)
    crown,cr=build_cage(crown_controls(v),origin=CROWN_ORIGIN,step=CROWN_STEP,shape=CROWN_SHAPE)
    nose,nr=build_cage(nose_controls(v),origin=NOSE_ORIGIN,step=NOSE_STEP,shape=NOSE_SHAPE)
    delta=evaluate_displacement(q,{'lower':lower,'crown':crown,'nose':nose})
    out=v.copy();out[skin]+=delta
    if not np.array_equal(out[eye],v[eye]):raise ValueError('Independent eyes moved')
    return out,{'lower_face':lr,'crown':cr,'nose':nr,'maximum_motion':float(np.linalg.norm(delta,axis=1).max()),
                'independent_eyes_unchanged':True,'scope':'Original full-region design candidate; actual views and validity sampling required'}, {'lower':lower,'crown':crown,'nose':nose}
