"""Explicit 1-mm occupied-site engineering contact, NOT render-boundary certification."""
from pathlib import Path
import hashlib, itertools, json, sys, time
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, diags
HERE=Path(__file__).resolve().parent
for p in (HERE,HERE.parent/'mechanics'):sys.path.insert(0,str(p))
from volume_guard import certify_positive_volume
COARSE_REQUIRED_GUARDS=('coarse_physical_site_initial_state','coarse_native_site_contact','positive_carrier_volume','bounded_physical_site_motion','coarse_site_path_checks','finite_source_origin_attachments')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def occupied_site_map(solid,points,H=6.):
    """Exact regular-carrier lookup; no nearest-tet extrapolation."""
    X,T=solid.X,solid.T
    origin=np.floor(X.min(0)/H)*H
    grid=np.rint((X-origin)/H).astype(np.int64)
    if np.max(abs(origin+H*grid-X))>1e-10:raise ValueError('not regular carrier')
    chain=grid[T];order=np.argsort(chain.sum(2),axis=1)
    chain=np.take_along_axis(chain,order[:,:,None],axis=1)
    axes=np.argmax(np.diff(chain,axis=1),axis=2)
    perms=np.array(list(itertools.permutations(range(3))))
    code=np.full(len(T),-1,int)
    for k,perm in enumerate(perms):code[np.all(axes==perm,axis=1)]=k
    if np.any(code<0):raise ValueError('non-Freudenthal carrier')
    keys=np.c_[chain[:,0],code].astype(np.int64)
    pgrid=(points-origin)/H;c=np.floor(pgrid).astype(np.int64);f=pgrid-c
    po=np.argsort(-f,axis=1,kind='stable');pc=np.full(len(points),-1,int)
    for k,perm in enumerate(perms):pc[np.all(po==perm,axis=1)]=k
    pkeys=np.c_[c,pc].astype(np.int64)
    dtype=np.dtype([(str(i),np.int64) for i in range(4)])
    kv=np.ascontiguousarray(keys).view(dtype).ravel();pv=np.ascontiguousarray(pkeys).view(dtype).ravel()
    sort=np.argsort(kv);loc=np.searchsorted(kv[sort],pv)
    if np.any(loc>=len(T)):raise ValueError('occupied site outside carrier')
    parent=sort[loc]
    if np.any(kv[parent]!=pv):raise ValueError('occupied site missing actual parent')
    w=np.einsum('nij,nj->ni',solid.inv[parent],points-X[T[parent,0]])
    w=np.c_[1-w.sum(1),w]
    if w.min() < -1e-12:raise ValueError('site extrapolation forbidden')
    w[np.abs(w)<1e-14]=0
    C=csr_matrix((w.ravel(),(np.repeat(np.arange(len(points)),4),T[parent].ravel())),shape=(len(points),len(X)))
    C.eliminate_zeros()
    if np.max(abs(C@X-points))>1e-9:raise ValueError('site map mismatch')
    return C,parent

from compact_penetration_barrier import compact_barrier_terms

class CoarsePhysicalSiteContact:
    guard_scope='coarse_physical_sites'
    fixture_only=False
    def __init__(self,solid,C,weights,rigid,*,initial_occupancy_verified,attachment_qualified,
                 clearance_mm=.1,penalty_N_mm4=.01,max_penetration_mm=.05,
                 maximum_reference_motion_mm=np.inf,maximum_increment_mm=.5,path_spacing_mm=.25,
                 jacobian_floor=.1,provenance=None,reference_lower_bound=None,barrier_band_mm=.01,barrier_scale=1.):
        started=time.perf_counter();self.solid=solid;self.C=csr_matrix(C);self.weights=np.asarray(weights,float)
        self.rigid=rigid;self.P0=self.C@solid.X;self.h=float(clearance_mm);self.k=float(penalty_N_mm4)
        self.penetration_limit=float(max_penetration_mm);self.reference_limit=float(maximum_reference_motion_mm)
        self.barrier_band=float(barrier_band_mm);self.barrier_scale=float(barrier_scale)
        if not 0<self.barrier_band<=self.penetration_limit or not np.isfinite(self.barrier_scale) or self.barrier_scale<=0:raise ValueError("positive barrier band/scale required")
        self.increment_limit=float(maximum_increment_mm);self.path_spacing=float(path_spacing_mm);self.jacobian_floor=float(jacobian_floor)
        if np.any(self.weights<=0) or self.weights.shape!=(self.C.shape[0],):raise ValueError('positive occupied site volumes required')
        self.lower=np.asarray(rigid.distance_lower_bound(self.P0) if reference_lower_bound is None else reference_lower_bound,float)
        self.cache=None;self.state_cache=[];self.query_count=0;self.query_seconds=0.
        reference=self.endpoint(solid.X)
        self.certification={k:True for k in COARSE_REQUIRED_GUARDS}
        self.certification['coarse_physical_site_initial_state']=bool(initial_occupancy_verified and reference['feasible'])
        self.certification['finite_source_origin_attachments']=bool(attachment_qualified)
        self.provenance=dict(provenance or {})
        self.provenance.update(guard_scope=self.guard_scope,contact_enforcement='compact_cap_barrier_v1',compact_barrier_source_sha256=sha(HERE/'compact_penetration_barrier.py'),contact_model='positive occupied voxel-center quadrature; sum of each independently active finite rigid-owner contact; no self-surface pair energy',query_revision='complete-band-owner-sum-interior-point-v3',
            physical_contact_sites=len(self.P0),site_volume_mm3_min=float(self.weights.min()),site_spacing_mm=solid.mm_per_L0,
            clearance_mm=self.h,penalty_stiffness_N_per_mm4=self.k,energy='0.5*k*sum_site,sum_owner(V_site*max(clearance-signed_finite_owner_gap,0)^2)',
            contact_tangent='PSD normal Gauss-Newton penalty plus mu/slack^2 normal rank one; signed-gap curvature omitted, not exact Hessian',
            penetration_constraint='signed_gap+maximum_allowed_site_penetration>0, independently per rigid owner',
            numerical_barrier=dict(band_mm=self.barrier_band,initial_scale=self.barrier_scale,baseline_mu_min_N_mm=float(self.k*self.weights.min()*(self.h+self.penetration_limit)*self.barrier_band),baseline_mu_max_N_mm=float(self.k*self.weights.max()*(self.h+self.penetration_limit)*self.barrier_band),mu0='k*V_site*(clearance+penetration_limit)*band',energy='mu*(-log(slack/band)+slack/band-1) for 0<slack<band, otherwise0',reaction='lambda=mu*(1/slack-1/band); lambda*slack<=mu',physical_contact_coefficients_unchanged=True),maximum_allowed_site_penetration_mm=self.penetration_limit,
            maximum_reference_site_motion_mm=None if not np.isfinite(self.reference_limit) else self.reference_limit,maximum_trial_site_increment_mm=self.increment_limit,
            maximum_path_sample_spacing_mm=self.path_spacing,global_surface_collision_certified=False,self_collision_certified=False,
            voxel_center_to_corner_distance_mm=float(np.sqrt(3)/2*solid.mm_per_L0),
            limitation='1-L0 occupancy sampling with complete finite-feature queries inside the contact band; boundary crossings, same-owner medial-axis nonsmoothness and sub-site collisions may occur. Independent owner penalties can count overlapping fixed-owner surfaces separately. No globally collision-free CAD claim.',
            rigid_query=rigid.report,initial_site_state=reference,setup_seconds=time.perf_counter()-started)
        self.report=self.provenance
    def set_barrier_scale(self,scale):
        scale=float(scale)
        if not np.isfinite(scale) or scale<=0:raise ValueError('positive finite barrier scale required')
        if scale!=self.barrier_scale:
            self.barrier_scale=scale;self.cache=None;self.state_cache=[]
    def _state(self,q):
        q=np.asarray(q,float)
        for index, cached in enumerate(self.state_cache):
            if np.array_equal(q,cached['q']):
                self.cache=self.state_cache.pop(index);self.state_cache.append(self.cache);return self.cache
        t=time.perf_counter();P=self.C@q;motion=np.linalg.norm(P-self.P0,axis=1)
        lower=self.lower-motion;ids=np.flatnonzero(lower<=self.h)
        if len(ids) and hasattr(self.rigid,'query_band'):
            result=self.rigid.query_band(P[ids],clearance=self.h,penetration_limit=self.penetration_limit,owner_terms=True)
        else:
            result=self.rigid.query(P[ids]) if len(ids) else dict(signed_gap=np.empty(0),normal=np.empty((0,3)),valid=np.empty(0,bool))
        gap=np.asarray(result['signed_gap']);normal=np.asarray(result['normal']);valid=np.asarray(result['valid'])
        force_ids=ids[np.asarray(result.get('contact_site_index',np.arange(len(ids))),int)]
        force_gap=np.asarray(result.get('contact_signed_gap',gap))
        normal=np.asarray(result.get('contact_normal',normal))
        depth=np.maximum(self.h-force_gap,0);gradient=-(self.k*self.weights[force_ids]*depth)[:,None]*normal
        penalty_gradient=np.asarray(self.C[force_ids].T@gradient)
        mu=self.barrier_scale*self.k*self.weights[force_ids]*(self.h+self.penetration_limit)*self.barrier_band
        barrier=compact_barrier_terms(force_gap,mu,penetration_limit=self.penetration_limit,band=self.barrier_band)
        reaction_force=np.asarray(self.C[force_ids].T@(barrier['reaction'][:,None]*normal))
        penalty_energy=float(.5*self.k*np.dot(self.weights[force_ids],depth**2))
        barrier_energy=float(np.sum(barrier['energy']))
        g=penalty_gradient-reaction_force
        if not barrier['valid'].all() or not valid.all():g=np.full_like(q,np.nan);barrier_energy=np.inf
        elapsed=time.perf_counter()-t;self.query_count+=1;self.query_seconds+=elapsed
        s=dict(q=q.copy(),points=P,motion=motion,ids=ids,gap=gap,normal=normal,valid=valid,
            force_ids=force_ids,force_gap=force_gap,force_owner=np.asarray(result.get('contact_owner_id',result.get('owner_id',np.full(len(force_ids),-1))),int),active=depth>0,
            energy=penalty_energy+barrier_energy,gradient=np.asarray(g),penalty_energy=penalty_energy,penalty_gradient=penalty_gradient,
            barrier_energy=barrier_energy,barrier_gradient=-reaction_force,reaction_force=reaction_force,barrier=barrier,mu=mu,
            seconds=elapsed,max_penetration=float(np.maximum(-gap,0).max(initial=0)),
            maximum_closed_owner_penetration_mm=float(result.get('maximum_closed_owner_penetration_mm',np.maximum(-gap,0).max(initial=0))),
            minimum_queried_gap=float(gap.min(initial=np.inf)),minimum_unqueried_gap_bound=float(lower[lower>self.h].min(initial=np.inf)))
        self.cache=s;self.state_cache.append(s)
        if len(self.state_cache)>4:self.state_cache.pop(0)
        return s
    def endpoint(self,q):
        J=np.linalg.det(self.solid.deformation(q))
        if not np.isfinite(J).all() or J.min()<=self.jacobian_floor:return dict(feasible=False,reason='positive carrier J',minimum_J=float(J.min()))
        s=self._state(q);gross=max(s['max_penetration'],s['maximum_closed_owner_penetration_mm'])
        good=bool(s['valid'].all() and np.isfinite(s['energy']) and s['motion'].max(initial=0)<=self.reference_limit+1e-12 and gross<self.penetration_limit and s['barrier']['valid'].all())
        return dict(feasible=good,guard_scope=self.guard_scope,minimum_J=float(J.min()),physical_sites=len(self.P0),queried_sites=len(s['ids']),
            active_contact_sites=int(len(np.unique(s['force_ids'][s['active']]))),active_owner_contact_terms=int(s['active'].sum()),minimum_queried_finite_gap_mm=s['minimum_queried_gap'],
            minimum_unqueried_gap_lower_bound_mm=s['minimum_unqueried_gap_bound'],maximum_site_penetration_mm=gross,
            invalid_finite_queries=int((~s['valid']).sum()),maximum_site_motion_from_reference_mm=float(s['motion'].max(initial=0)),
            site_query_seconds=s['seconds'],barrier_scale=self.barrier_scale,minimum_penetration_slack_mm=float(s['barrier']['slack'].min(initial=np.inf)),active_constraint_reactions=int(s['barrier']['active'].sum()),global_surface_collision_certified=False)
    def energy(self,q):
        s=self._state(q);return s['energy'],s['gradient']
    def tangent(self,q):
        s=self._state(q);take=s['active'];ids=s['force_ids'][take];A=self.C[ids].tocoo();normal=s['normal'][take]
        r=np.repeat(A.row,3);c=(3*A.col[:,None]+np.arange(3)).ravel();data=(A.data[:,None]*normal[A.row]).ravel()
        B=coo_matrix((data,(r,c)),shape=(len(ids),3*len(self.solid.X))).tocsr()
        return (B.T@diags(self.k*self.weights[ids]+s['barrier']['curvature'][take])@B).tocsr()
    def physical_site_positions(self,q):
        return self.C@np.asarray(q,float)
    def max_physical_site_difference(self,q1,q2):
        return float(np.linalg.norm(self.C@(np.asarray(q1)-np.asarray(q2)),axis=1).max(initial=0))
    def contact_accounting(self,q,include_vectors=True,include_terms=False):
        state=self._state(q);b=state['barrier'];norm=lambda x:float(np.max(np.linalg.norm(x,axis=1),initial=0))
        result=dict(barrier_scale=self.barrier_scale,quadratic_contact_energy_N_mm=state['penalty_energy'],
            numerical_barrier_energy_N_mm=state['barrier_energy'],active_constraint_reactions=int(b['active'].sum()),
            maximum_constraint_reaction_N=float(b['reaction'].max(initial=0)),
            maximum_nodal_constraint_reaction_N=norm(state['reaction_force']),
            minimum_penetration_slack_mm=None if not len(b['slack']) else float(b['slack'].min()),
            maximum_complementarity_N_mm=float(b['complementarity'].max(initial=0)),
            maximum_active_mu_N_mm=float(state['mu'][b['active']].max(initial=0)),
            sum_complementarity_N_mm=float(np.sum(b['complementarity'])),
            complementarity_bound_satisfied=bool(np.all(b['complementarity']<=state['mu']*(1+1e-12))),
            constraint_reaction_net_N=state['reaction_force'].sum(0).tolist(),
            quadratic_contact_force_net_N=(-state['penalty_gradient']).sum(0).tolist(),
            tangent_scope='PSD normal approximation, not exact Hessian',
            open_sheet_barrier_scope='Unsigned gap>=0 implies slack>=penetration_limit>band; no open-sheet barrier reaction')
        result.update(penalty_energy_N_mm=state['penalty_energy'],barrier_energy_N_mm=state['barrier_energy'],
            minimum_slack_mm=result['minimum_penetration_slack_mm'],max_lambda_s_N_mm=result['maximum_complementarity_N_mm'],
            sum_lambda_s_N_mm=result['sum_complementarity_N_mm'],active_barrier_owner_terms=result['active_constraint_reactions'])
        result['per_owner']=[]
        for owner in np.unique(state['force_owner']):
            selected=state['force_owner']==owner
            result['per_owner'].append(dict(owner_id=int(owner),terms=int(selected.sum()),active_barrier_terms=int(b['active'][selected].sum()),
                minimum_slack_mm=float(b['slack'][selected].min()),maximum_lambda_N=float(b['reaction'][selected].max(initial=0)),
                max_lambda_s_N_mm=float(b['complementarity'][selected].max(initial=0)),sum_lambda_s_N_mm=float(b['complementarity'][selected].sum()),
                reaction_net_N=np.sum(b['reaction'][selected,None]*state['normal'][selected],axis=0).tolist()))
        if include_vectors:
            result.update(penalty_gradient_N=state['penalty_gradient'],barrier_gradient_N=state['barrier_gradient'])
        if include_terms:
            result.update(site_indices=state['force_ids'],owner_ids=state['force_owner'],lambda_N=b['reaction'],slack_mm=b['slack'],mu_N_mm=state['mu'])
        return result
    def safe_step_bound(self,q,d,safety=.9):
        state=self._state(q)
        if not state['barrier']['valid'].all():return 0.
        P=self.C@q;D=self.C@d;length=np.linalg.norm(D,axis=1);maximum=length.max(initial=0)
        if maximum==0:return 1.
        left=self.reference_limit-np.linalg.norm(P-self.P0,axis=1);moving=length>0
        derivative=np.einsum('ij,ij->i',state['normal'],D[state['force_ids']]);closing=derivative<0
        fraction=1. if not closing.any() else float(np.min(safety*state['barrier']['slack'][closing]/(-derivative[closing])))
        return max(0.,min(1.,fraction,safety*self.increment_limit/maximum,float(np.min(safety*left[moving]/length[moving]))))
    def path(self,q0,q1):
        volume=certify_positive_volume(self.solid,q0,q1,floor=self.jacobian_floor)
        increment=float(np.linalg.norm(self.C@(q1-q0),axis=1).max(initial=0))
        if not volume['certified']:return dict(feasible=False,coarse_whole_segment_checked=False,reason='positive-J path',volume=volume)
        count=max(1,int(np.ceil(increment/self.path_spacing)));states=[]
        if count>128:return dict(feasible=False,coarse_whole_segment_checked=False,reason='bounded path sampling budget exceeded')
        for j in range(count+1):
            e=self.endpoint(q0+(j/count)*(q1-q0));states.append(e)
            if not e['feasible']:return dict(feasible=False,coarse_whole_segment_checked=False,samples=states,volume=volume)
        return dict(feasible=True,coarse_whole_segment_checked=True,scope='finite occupied-site samples, not continuous surface collision certification',
            samples=len(states),maximum_site_increment_mm=increment,maximum_sample_spacing_mm=increment/count,
            maximum_sampled_penetration_mm=max(s['maximum_site_penetration_mm'] for s in states),volume=volume)

def create_guard(solid,constraints):
    from coarse_rigid_query import NativeRigidQuery
    geo=HERE.parent/'geometry';cells_path=geo/'whole-face-integration-cells.npz'
    with np.load(cells_path) as z:
        ijk=z['cell_ijk'].astype(int);spacing=float(z['grid_spacing'])*solid.mm_per_L0
        points=(z['grid_origin']+(ijk+.5)*float(z['grid_spacing']))*solid.mm_per_L0
        native_sha=str(z['source_sha256'])
    if Path(solid.provenance['domain_path']).name!='whole-face-continuum-final.npz':raise ValueError('coarse voxel adapter requires the disclosed original domain')
    with np.load(geo/'source-membership.npz') as z:initial_overlap=int(z['rigid'][tuple(ijk.T)].sum())
    if initial_overlap:raise ValueError('retained physical sites overlap original native occupancy field')
    C,parent=occupied_site_map(solid,points,H=6*solid.mm_per_L0)
    rigid=NativeRigidQuery.from_native(candidate_count=48,coordinate_scale=solid.mm_per_L0)
    if native_sha!=solid.provenance['source_registered_anatomy_sha256']:raise ValueError('site source hash mismatch')
    cache_path=HERE/'coarse-site-reference-band-v1.npz';site_sha=sha(cells_path);lower=None
    if cache_path.exists():
        with np.load(cache_path) as z:
            if str(z['site_sha256'])==site_sha and str(z['native_sha256'])==native_sha:
                lower=z['lower_bound_mm']
    if lower is None:
        print('Building tight finite reference site band once',flush=True)
        lower=rigid.clipped_distance_lower_bound(points,2.5*solid.mm_per_L0)
        np.savez_compressed(cache_path,lower_bound_mm=lower,site_sha256=np.array(site_sha),native_sha256=np.array(native_sha))
    support_path=HERE.parent/'mechanics/coarse-physical-site-v1/supports/clipped-origin-support.json'
    support=json.loads(support_path.read_text())
    attachment=support.get('finite_material_attachment_qualified') is True and support.get('constraint_source')==constraints.report['source']
    return CoarsePhysicalSiteContact(solid,C,np.full(len(points),spacing**3),rigid,
        initial_occupancy_verified=initial_overlap==0,attachment_qualified=attachment,reference_lower_bound=lower,
        provenance=dict(source_domain_sha256=solid.provenance['domain_sha256'],source_cells_sha256=site_sha,
        reference_band_cache_sha256=sha(cache_path),reference_distance_bound_clip_mm=2.5*solid.mm_per_L0,
        reference_motion_policy='No artificial physical displacement cap; clipped lower bound stays valid beyond its band but queries increase. Per-trial and path sample bounds remain enforced.',
        native_source_sha256=native_sha,rigid_query_source_sha256=sha(HERE/'coarse_rigid_query.py'),initial_source_field_overlap_sites=initial_overlap,
        occupancy_semantics='Original bounded native z-ray center-membership field; open-owner caveats retained, not a closed-solid claim.'))
