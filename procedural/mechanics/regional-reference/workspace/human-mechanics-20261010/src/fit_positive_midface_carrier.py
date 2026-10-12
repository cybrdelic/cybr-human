"""Fit the authored inner sheet from a positive thin initialization.
The native exterior, oral interface and omitted-structure collars stay fixed.
Every accepted line search is positive. This is geometric mesh authoring, not
an equilibrium solve or a substitute for surface-overlap/contact checks.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from scipy.optimize import minimize
p=argparse.ArgumentParser();p.add_argument('--seed',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=240);p.add_argument('--norm-caps',type=Path);p.add_argument('--quality-weight',type=float,default=0.);a=p.parse_args()
s=np.load(a.seed);q=np.load(a.target);N=len(s['midface_column_nodes']);grid=s['midface_column_nodes'];x0=s['vertices'].copy();target=q['inner_carrier_displacement'];u0=s['inner_carrier_displacement'];top=x0[grid[:,0]].copy();t=s['tetrahedra'];part=s['face_parts'][s['cell_parameters'][:,0]];active=np.flatnonzero(part!=20);tt=t[active];assert tt.max()<5*N
fixed=np.unique(np.r_[s['protected_boundary_columns'],s['oral_interface_columns']]);free=np.setdiff1d(np.arange(N),fixed);u=u0.copy();frac=np.arange(5)/4;col=tt%N;f=tt//N/4
assert np.array_equal(s['tetrahedra'],q['tetrahedra']) and np.array_equal(s['outer_column_native_ids'],q['outer_column_native_ids'])
def detgrad(xx):
 e=xx[tt[:,1:]]-xx[tt[:,0,None]];g=np.stack([np.cross(e[:,1],e[:,2]),np.cross(e[:,2],e[:,0]),np.cross(e[:,0],e[:,1])],axis=1);d=np.einsum('ij,ij->i',e[:,0],g[:,0]);return d,np.concatenate([-g.sum(1)[:,None,:],g],axis=1)
d0,_=detgrad(x0);assert d0.min()>0
threshold=d0*1e-5;history=[];start=time.time();invalid=0
caps=np.load(a.norm_caps)['column_norm_caps'] if a.norm_caps else np.full(N,np.inf)
assert (np.linalg.norm(u0,axis=1)<caps).all()
uncapped_target_percentiles=np.percentile(np.linalg.norm(target,axis=1),[0,25,50,75,100]).tolist()
if a.norm_caps:target=target*np.minimum(1,.92*caps/np.maximum(np.linalg.norm(target,axis=1),1e-30))[:,None]
# The barrier is integrated by rest surface cell rather than tetrahedron volume:
# tiny corner elements must not be free to invert because their volume is small.
def objective(z,weight):
 global invalid
 u[free]=z.reshape(-1,3);x=x0.copy();x[:5*N]=np.concatenate([top+v*u for v in frac]);d,dg=detgrad(x)
 slack=caps*caps-np.sum(u*u,axis=1)
 if np.any(d<=threshold) or np.any(slack<=0):
  invalid+=1;return 1e30,np.zeros_like(z)
 delta=u-target;energy=.5*np.mean(np.sum(delta[free]**2,axis=1));grad=delta/len(free);energy-=weight*np.mean(np.log(d/d0));dd=dg.copy();dg*=(-weight/len(tt)/d)[:,None,None]
 if a.quality_weight:
  ee=x[tt[:,1:]]-x[tt[:,0,None]];norm2=np.einsum('tij,tij->t',ee,ee);power=d**(-2/3);shape=norm2*power;energy+=a.quality_weight*np.mean(shape)
  qg=np.concatenate([-2*ee.sum(1)[:,None,:],2*ee],axis=1)*power[:,None,None]-(2/3)*(shape/d)[:,None,None]*dd
  dg+=a.quality_weight/len(tt)*qg
 if a.norm_caps:
  energy-=.05*np.mean(np.log(slack[free]/(caps[free]*caps[free])));grad[free]+=.10/len(free)*u[free]/slack[free,None]
 for k in range(4):np.add.at(grad,col[:,k],dg[:,k]*f[:,k,None])
 return float(energy),grad[free].ravel()
z=u0[free].ravel().copy();results=[]
energy,gcheck=objective(z,.5);checks=[]
for k in [11,len(z)//3,len(z)//2,len(z)-15]:
 dz=np.zeros_like(z);dz[k]=1e-6;fd=(objective(z+dz,.5)[0]-objective(z-dz,.5)[0])/2e-6;error=abs(fd-gcheck[k]);checks.append({'coordinate':k,'analytic':float(gcheck[k]),'finite_difference':float(fd),'absolute_error':float(error)})
assert max(q['absolute_error'] for q in checks)<1e-6
for weight in [.5,.15]:
 def callback(z):
  u[free]=z.reshape(-1,3);x=x0.copy();x[:5*N]=np.concatenate([top+v*u for v in frac]);d,_=detgrad(x);history.append({'stage_weight':weight,'min_det':float(d.min()),'median_depth':float(np.median(np.linalg.norm(u,axis=1))),'objective':float(objective(z,weight)[0])})
 memory=[];e,g=objective(z,weight);accepted=0;reason='iteration budget';evaluations=1
 for it in range(a.iterations):
  if np.max(abs(g))<1e-6:reason='gradient tolerance';break
  direction=g.copy();alphas=[]
  for ss,yy,rho in reversed(memory):
   al=rho*np.dot(ss,direction);alphas.append(al);direction-=al*yy
  gamma=np.dot(memory[-1][0],memory[-1][1])/np.dot(memory[-1][1],memory[-1][1]) if memory else len(free)*.2
  direction*=gamma
  for (ss,yy,rho),al in zip(memory,reversed(alphas)):
   direction+=ss*(al-rho*np.dot(yy,direction))
  direction=-direction
  slope=float(np.dot(g,direction))
  if slope>=0:memory=[];direction=-g*len(free)*.2;slope=float(np.dot(g,direction))
  alpha=1.;found=False
  for ls in range(45):
   nz=z+alpha*direction;ne,ng=objective(nz,weight);evaluations+=1
   if ne<1e29 and ne<=e+1e-4*alpha*slope:found=True;break
   alpha*=.5
  if not found:reason='line search cannot make positive progress';break
  ss=nz-z;yy=ng-g;sy=float(np.dot(ss,yy))
  if sy>1e-12*np.linalg.norm(ss)*np.linalg.norm(yy):memory.append((ss,yy,1/sy));memory=memory[-12:]
  z=nz;e=ne;g=ng;accepted+=1
  if it%20==0:callback(z);print(json.dumps({'iteration':it,'weight':weight,'objective':e,'gradient_inf':float(np.max(abs(g))),'step':alpha}),flush=True)
  if np.max(abs(ss))<1e-10:reason='step tolerance';break
 callback(z);results.append({'weight':weight,'success':reason=='gradient tolerance','message':reason,'iterations':accepted,'function_evaluations':evaluations,'gradient_inf':float(np.max(abs(g)))})

u[free]=z.reshape(-1,3);x=x0.copy();x[:5*N]=np.concatenate([top+v*u for v in frac]);dm=np.stack([x[t[:,i]]-x[t[:,0]] for i in [1,2,3]],-1);det=np.linalg.det(dm);cond=np.linalg.cond(dm);passed=bool((det>0).all());a.output.mkdir(parents=True,exist_ok=False);arrays={k:s[k] for k in s.files};arrays['vertices']=x;arrays['inner_carrier_displacement']=u.copy();np.savez_compressed(a.output/'rest_geometry.npz',**arrays)
report={'schema':'positive-carrier-authoring-v1','status':'geometric candidate only; no overlap/contact/mechanics acceptance','all_tets_positive':passed,'negative_tets':int((det<=0).sum()),'min_det':float(det.min()),'max_condition':float(cond.max()),'condition_percentiles':np.percentile(cond,[50,90,99]).tolist(),'stage_results':results,'invalid_trial_rejections':invalid,'seconds':time.time()-start,'target_depth_percentiles':np.percentile(np.linalg.norm(target,axis=1),[0,25,50,75,100]).tolist(),'fitted_depth_percentiles':np.percentile(np.linalg.norm(u,axis=1),[0,25,50,75,100]).tolist(),'uncapped_target_depth_percentiles':uncapped_target_percentiles,'bone_cap_target_fraction':.92 if a.norm_caps else None,'quality_weight':a.quality_weight,'gradient_checks':checks,'minimum_column_cap_slack':float(np.min(caps-np.linalg.norm(u,axis=1))) if a.norm_caps else None,'fixed_column_error':float(np.max(abs(u[fixed]-u0[fixed]))),'exact_outer_coordinates':bool(np.array_equal(x[grid[:,0]],top)),'source_seed_sha256':hashlib.sha256(a.seed.read_bytes()).hexdigest(),'source_target_sha256':hashlib.sha256(a.target.read_bytes()).hexdigest(),'history':history}
(a.output/'geometry_fit_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)
if not passed:raise SystemExit(1)
