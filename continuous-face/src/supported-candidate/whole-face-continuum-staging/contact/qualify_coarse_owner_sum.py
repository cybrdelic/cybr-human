import sys,json,hashlib
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from scipy.sparse import csr_matrix
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from coarse_rigid_query import NativeRigidQuery
from coarse_physical_site_contact import CoarsePhysicalSiteContact

def energy(r,p):
 z=r.query_band(np.asarray(p).reshape(1,3),owner_terms=True);assert z['valid'].all();d=np.maximum(.1-z['contact_signed_gap'],0);return .005*np.dot(d,d),-.01*np.sum(d[:,None]*z['contact_normal'],axis=0),z
r=NativeRigidQuery.from_native();data=json.load(open(P/'coarse-hotspot-feature-diagnosis.json'));hot=next(x for x in data['rows'] if x['site']==342284);points=np.array(hot['reference'])+hot['carrier_weight']*np.array(data['displacements_mm']);fd_error=0.
for p in points:
 e,g,z=energy(r,p)
 for axis in range(3):
  d=np.eye(3)[axis]*1e-6;fd=(energy(r,p+d)[0]-energy(r,p-d)[0])/2e-6;fd_error=max(fd_error,abs(fd-g[axis]))
# The measured endpoints have opposite nearest owners; resolve their bisector.
a,b=points[2].copy(),points[3].copy()
def difference(p):
 z=energy(r,p)[2];gap=dict(zip(z['contact_owner_id'],z['contact_signed_gap']));return gap[5]-gap[6]
assert difference(a)*difference(b)<0
for _ in range(40):
 m=(a+b)/2
 if difference(a)*difference(m)<=0:b=m
 else:a=m
mid=(a+b)/2;direction=(points[3]-points[2]);direction/=np.linalg.norm(direction);eps=1e-7
em,gm,zm=energy(r,mid-eps*direction);ep,gp,zp=energy(r,mid+eps*direction)
force_jump=float(np.linalg.norm(gp-gm));oldm=r.query_band((mid-eps*direction)[None]);oldp=r.query_band((mid+eps*direction)[None]);old_force_jump=float(np.linalg.norm(-.01*(.1-oldp['signed_gap'][0])*oldp['normal'][0]+.01*(.1-oldm['signed_gap'][0])*oldm['normal'][0]))
assert force_jump<1e-8 and fd_error<1e-9
# Symmetric two-sheet model checks the actual adapter's sum and PSD tangent.
X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float);T=np.array([[0,1,2,3]]);s=SimpleNamespace(X=X,T=T,inv=np.eye(3)[None],mm_per_L0=1.);s.deformation=lambda q:np.stack([q[T[:,k]]-q[T[:,0]] for k in (1,2,3)],axis=2)
V=np.array([[-3,-3,.24],[4,-3,.24],[-3,4,.24],[-3,-3,.26],[4,-3,.26],[-3,4,.26]])
rigid=NativeRigidQuery(V,[[0,1,2],[3,4,5]],[1,2]);guard=CoarsePhysicalSiteContact(s,csr_matrix([[.25]*4]),[1.],rigid,initial_occupancy_verified=True,attachment_qualified=True)
e,g=guard.energy(X);assert np.max(abs(g))<1e-14;assert abs(e-.01*.09**2)<1e-15;assert guard.endpoint(X)['active_owner_contact_terms']==2
rng=np.random.default_rng(8);d=rng.normal(size=X.shape);q=X+[0,0,.002];ee,gg=guard.energy(q);fd=(guard.energy(q+1e-6*d)[0]-guard.energy(q-1e-6*d)[0])/2e-6;adapter_error=abs(fd-np.sum(gg*d));mineig=float(np.linalg.eigvalsh(guard.tangent(q).toarray()).min());assert adapter_error<1e-10 and mineig>-1e-12
out=dict(passed=True,scope='Bounded owner-switch and derivative regression, not a transient or global collision test',native_hotspot_site=342284,owners=[5,6],tested_native_positions=len(points),native_energy_gradient_max_error=fd_error,owner_bisector_midpoint_mm=mid.tolist(),symmetric_probe_spacing_mm=2*eps,single_nearest_force_jump_N=old_force_jump,summed_owner_force_difference_N=force_jump,synthetic_midpoint_net_force_max_N=float(np.max(abs(g))),synthetic_adapter_gradient_error=adapter_error,synthetic_tangent_min_eigenvalue=mineig,unchanged_parameters=dict(clearance_mm=.1,stiffness_N_per_mm4=.01,penetration_limit_mm=.05,carrier_J_floor=.1,trial_increment_mm=.5,path_spacing_mm=.25),source_sha256={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['coarse_rigid_query.py','coarse_physical_site_contact.py']})
(P/'coarse-owner-sum-qualification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
