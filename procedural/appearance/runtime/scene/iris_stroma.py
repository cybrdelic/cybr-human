"""Original parametric iris relief and pigment proxies; no images or fitted subject data.
Morphological vocabulary only: https://doi.org/10.1098/rsos.150424 and
https://www.sci.utah.edu/~lefohn/work/eye/irisSynthesis.pdf
Dimensions, distribution and colors below are artistic and uncalibrated.
"""
import numpy as np
SEED=2228

def configure(seed):
    global SEED
    SEED=seed

def fields(r,t):
    u=np.clip((r-2.08)/3.42,0,1);phase=(SEED%101)*.027
    tw=t+.015*np.sin(13*t+phase)*u+.009*np.sin(29*t-phase)
    fine=.5+.22*np.sin(47*tw+u*6)+.17*np.sin(109*tw+2*np.sin(23*tw)-u*10)+.06*np.sin(19*tw-u*20)
    rc=3.15+.14*np.sin(7*t+phase)+.07*np.sin(19*t-phase)+.035*np.sin(43*t)
    ridge=np.exp(-((r-rc)/.115)**2)
    rng=np.random.default_rng(SEED);crypt=np.zeros_like(r)
    for k in range(24):
        angle=rng.uniform(0,2*np.pi);r0=3.35+rng.uniform(-.20,.80);wr=rng.uniform(.14,.34);wt=rng.uniform(.027,.055)
        dt=np.arctan2(np.sin(t-angle),np.cos(t-angle));ell=((r-r0)/wr)**2+(dt/wt)**2
        crypt=np.maximum(crypt,np.exp(-ell)*rng.uniform(.55,1.))
    furrow=np.sin(37*u+1.4*np.sin(9*t))*np.clip((u-.55)/.3,0,1)
    return u,fine,ridge,crypt,furrow

def height(r,t):
    u,fine,ridge,crypt,furrow=fields(r,t)
    # Pupillary margin anterior to outer root, unlike the old reverse-sloped control.
    return 10.68-.40*u+.025*(fine-.5)+.070*ridge-.065*crypt-.010*furrow

def color(r,t):
    u,fine,ridge,crypt,furrow=fields(r,t)
    rgb=np.empty((*r.shape,3));rgb[...,0]=.080+.125*fine;rgb[...,1]=.057+.095*fine;rgb[...,2]=.028+.057*fine
    rgb+=ridge[...,None]*np.array([.034,.020,.006])
    rgb*=1-.38*crypt[...,None]
    rgb*=1+.075*furrow[...,None]
    limbus=.78*np.clip((r-5.03)/.43,0,1);ruff=.64*np.exp(-((r-2.10)/.13)**2)
    rgb*=1-np.maximum(limbus,ruff)[...,None]
    return np.concatenate([np.clip(rgb,0,1),np.ones((*r.shape,1))],axis=-1)

def audit(points,c,R):
    q=points-c;rr=np.linalg.norm(q[:,:2],axis=1);zz=q[:,2]
    front=np.sqrt(R*R-rr*rr)+.035+.36*np.maximum(0,1-(rr/5.56)**2)**2
    posterior=front-(.25+.20*np.maximum(0,1-(rr/5.56)**2))
    rear=zz-9.8;gap=posterior-zz;assert rear.min()>.01 and gap.min()>.01,(rear.min(),gap.min())
    tt=np.linspace(0,2*np.pi,512,endpoint=False);p=float(height(np.full_like(tt,2.08),tt).mean());o=float(height(np.full_like(tt,5.5),tt).mean());assert p>o
    return {'seed':SEED,'angular_samples':512,'radial_samples':80,'pupillary_margin_mean_z_relative':p,'outer_root_mean_z_relative':o,'pupillary_margin_anterior':True,'minimum_sampled_posterior_cornea_gap_design_units':float(gap.min()),'minimum_sampled_rear_plane_gap_design_units':float(rear.min()),'iris_depth_range_relative':[float(zz.min()),float(zz.max())],'shape':'Original conical stroma, irregular collarette, 24 soft crypt depressions, radial fiber relief and outer furrow proxy','limitations':'Authored dimensions/pigment distribution; no biological validation, motion model or microscopic layered scattering'}
