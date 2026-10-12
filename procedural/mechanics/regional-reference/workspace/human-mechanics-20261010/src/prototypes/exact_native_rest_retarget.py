"""Opt-in rest authoring from exact native IDs, with immobile finite bone roots.

This is mesh authoring, not a physical trajectory, growth law, or accepted solve.
No lost warp is reconstructed. Main source and saved cases are never changed.
The CLI only inspects anchors. A selected artistic pin and an explicit caller
are required to invoke the bounded interior solve and write a NEW case.

Global tissue overlap, dense skin self-contact, and static-bone admission are
separate mandatory gates. An output from this module is never mechanics-ready.
"""
from pathlib import Path
import argparse, copy, hashlib, json, sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from build_bilateral_cavity_case import read_arrays, write_arrays

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def cell_matrices(points,tets):
    return np.stack([points[tets[:,j]]-points[tets[:,0]] for j in (1,2,3)],axis=-1)

def exact_native_anchors(rest,binding,old_head,new_head,rigid_nodes,scale=.001):
    """Return material-identity Dirichlet data; never search spatial neighbors.

    Coordinates use solver units; head vertices use authored design units.
    The rigid set is only the finite root/collar, NOT old simulation cut pins.
    """
    rest=np.asarray(rest,float);old_head=np.asarray(old_head,float);new_head=np.asarray(new_head,float)
    if (not np.isfinite(scale) or scale<=0 or
        ('solver_units_per_design_unit' in binding and scale!=float(binding['solver_units_per_design_unit']))):
        raise ValueError('Anchor and binding coordinate scales differ')
    if old_head.shape!=new_head.shape or old_head.ndim!=2 or old_head.shape[1]!=3:
        raise ValueError('Native vertex layout differs')
    if (not np.isfinite(rest).all() or not np.isfinite(old_head).all() or
        not np.isfinite(new_head).all() or not np.isfinite(binding['points']).all()):
        raise ValueError('Nonfinite rest, head or binding coordinates')
    ids=np.asarray(binding['vertex_ids']);ix=np.asarray(binding['indices']);w=np.asarray(binding['weights'])
    if ix.shape!=w.shape or ix.shape!=(len(ids),3) or np.any(w<0) or not np.allclose(w.sum(1),1,rtol=0,atol=1e-13):
        raise ValueError('Invalid preserved binding layout/weights')
    if np.any(ix<0) or np.any(ix>=len(rest)) or np.any(ids<0) or np.any(ids>=len(old_head)):
        raise ValueError('Invalid material IDs')
    if np.max(abs(np.asarray(binding['points'])-old_head[ids]*scale),initial=0)>1e-13:
        raise ValueError('Binding is not on the pinned baseline head')
    rows=np.flatnonzero((w==1).any(1)&((w>0).sum(1)==1))
    nodes=ix[rows,np.argmax(w[rows],axis=1)].astype(np.int64)
    support=np.unique(ix[w>0]);unique,counts=np.unique(nodes,return_counts=True)
    if np.any(counts!=1) or not np.array_equal(unique,support):
        raise ValueError('Every effective boundary support must have one unique exact native anchor')
    if np.max(abs(rest[nodes]-old_head[ids[rows]]*scale),initial=0)>1e-13:
        raise ValueError('A surface anchor does not match its old exact native coordinate')
    rigid=np.unique(np.asarray(rigid_nodes,dtype=np.int64))
    if np.any(rigid<0) or np.any(rigid>=len(rest)):raise ValueError('Invalid rigid root/collar ID')
    if np.intersect1d(nodes,rigid).size:raise ValueError('Native boundary and rigid collar constraints overlap')
    target=rest.copy();target[nodes]=new_head[ids[rows]]*scale;target[rigid]=rest[rigid]
    fixed=np.unique(np.r_[nodes,rigid])
    return dict(nodes=fixed,targets=target,surface_nodes=nodes,native_ids=ids[rows],rigid_nodes=rigid,
        scope='Exact material-ID rest authoring. Old simulation cut pins may move during authoring and pin to the NEW rest during later mechanics.')

def segment_minimum_jacobian(F0,F1):
    """Minimum relative determinant on the complete linear nodal segment.

    A tetrahedron determinant is cubic in step fraction. Check its endpoints
    and all real stationary points, rather than sampling only the endpoint.
    Floating-point algebra is a numerical gate, not an interval proof.
    """
    d=F1-F0
    values=np.stack([np.linalg.det(F0+t*d) for t in (0.,1/3,2/3,1.)])
    vander=np.vander(np.array([0.,1/3,2/3,1.]),4,increasing=True)
    c=np.linalg.solve(vander,values)
    minimum=np.minimum(values[0],values[-1]);a=3*c[3];b=2*c[2];cc=c[1]
    quadratic=abs(a)>1e-12;disc=b*b-4*a*cc
    for sign in (-1,1):
        valid=quadratic&(disc>=0);z=np.zeros(len(a));z[valid]=(-b[valid]+sign*np.sqrt(disc[valid]))/(2*a[valid])
        take=valid&(z>0)&(z<1);q=((c[3]*z+c[2])*z+c[1])*z+c[0]
        minimum[take]=np.minimum(minimum[take],q[take])
    valid=(~quadratic)&(abs(b)>1e-12);z=np.zeros(len(a));z[valid]=-cc[valid]/b[valid]
    take=valid&(z>0)&(z<1);q=((c[3]*z+c[2])*z+c[1])*z+c[0]
    minimum[take]=np.minimum(minimum[take],q[take])
    return float(minimum.min())

class RestMeshObjective:
    """Uncalibrated distortion energy plus a numerical noninversion barrier.

    The default relative-J floor 0.2 is NOT an anatomical or mesh-quality
    acceptance threshold. Local J, conditioning and contact still require a
    separate comparison against the baseline before candidate admission.
    """
    def __init__(self,rest,tets,minimum_jacobian=.2,bulk=5.,barrier=.02):
        self.rest=np.asarray(rest,float);self.tets=np.asarray(tets,np.int64)
        self.dm=cell_matrices(self.rest,self.tets);det=np.linalg.det(self.dm)
        if np.any(det<=0):raise ValueError('Baseline contains a nonpositive cell')
        if not 0<minimum_jacobian<1:raise ValueError('Invalid relative Jacobian floor')
        self.inverse=np.linalg.inv(self.dm);self.grad=np.zeros((len(tets),4,3))
        self.grad[:,1:]=self.inverse;self.grad[:,0]=-self.grad[:,1:].sum(1)
        self.weights=det/det.sum();self.floor=minimum_jacobian;self.bulk=bulk;self.barrier=barrier
        # FEM scalar Dirichlet energy supplies only an initialization/preconditioner.
        local=np.einsum('tad,tbd,t->tab',self.grad,self.grad,self.weights)
        rows=np.repeat(self.tets,4,axis=1).ravel();cols=np.tile(self.tets,(1,4)).ravel()
        self.K=sparse.coo_matrix((local.ravel(),(rows,cols)),shape=(len(rest),len(rest))).tocsr()
    def deformation(self,x):return cell_matrices(x,self.tets)@self.inverse
    def evaluate(self,x,gradient=True):
        F=self.deformation(x);J=np.linalg.det(F)
        if not np.isfinite(J).all() or J.min()<=self.floor:return np.inf,None,F,J
        log=np.log(J);a=1-self.floor
        W=.5*(np.sum(F*F,axis=(1,2))-3)-log+.5*self.bulk*log*log
        W+=self.barrier*(-np.log((J-self.floor)/a)+(J-1)/a)
        E=float(self.weights@W)
        if not gradient:return E,None,F,J
        inverseT=np.linalg.inv(F).transpose(0,2,1)
        coeff=-1+self.bulk*log+self.barrier*J*(-1/(J-self.floor)+1/a)
        P=F+coeff[:,None,None]*inverseT
        local=np.einsum('tij,taj,t->tai',P,self.grad,self.weights)
        g=np.zeros_like(x)
        for a in range(4):np.add.at(g,self.tets[:,a],local[:,a])
        return E,g,F,J

def _relax(objective,x,fixed,max_iterations,tolerance):
    free=np.setdiff1d(np.arange(len(x)),fixed);history=[];pairs=[]
    if not len(free):return x,dict(converged=True,iterations=0,free_gradient_max=0.,history=[])
    precondition=np.repeat(1/np.maximum(objective.K.diagonal()[free]*(1+objective.bulk),1e-14),3)
    E,gradient,F,J=objective.evaluate(x)
    if not np.isfinite(E):return x,dict(converged=False,reason='infeasible-start')
    for iteration in range(max_iterations+1):
        g=gradient[free].ravel();residual=float(abs(g).max(initial=0))
        if residual<=tolerance:return x,dict(converged=True,iterations=iteration,free_gradient_max=residual,history=history)
        if iteration==max_iterations:break
        q=g.copy();alphas=[]
        for s,y,rho in reversed(pairs):
            alpha=rho*np.dot(s,q);alphas.append(alpha);q-=alpha*y
        direction=precondition*q
        for (s,y,rho),alpha in zip(pairs,reversed(alphas)):
            direction+=s*(alpha-rho*np.dot(y,direction))
        direction=-direction
        if not np.isfinite(direction).all() or np.dot(g,direction)>=0:
            pairs=[];direction=-precondition*g
        slope=float(g@direction);step=1.;accepted=False
        for _ in range(30):
            candidate=x.copy();candidate[free]+=step*direction.reshape(-1,3)
            En,gn,Fn,Jn=objective.evaluate(candidate)
            if np.isfinite(En) and En<=E+1e-4*step*slope and segment_minimum_jacobian(F,Fn)>objective.floor:
                accepted=True;break
            step*=.5
        if not accepted:return x,dict(converged=False,reason='no-feasible-descent',iterations=iteration,free_gradient_max=residual,history=history)
        s=(candidate[free]-x[free]).ravel();y=(gn[free]-gradient[free]).ravel();sy=float(s@y)
        if sy>1e-14*np.linalg.norm(s)*np.linalg.norm(y):pairs=(pairs+[(s,y,1/sy)])[-8:]
        history.append(dict(iteration=iteration,energy=En,minimum_relative_jacobian=float(Jn.min()),step=step))
        x,E,gradient,F,J=candidate,En,gn,Fn,Jn
    return x,dict(converged=False,reason='iteration-budget',iterations=max_iterations,free_gradient_max=residual,history=history)

def extend_rest(rest,tets,anchors,minimum_jacobian=.2,initial_fraction=.25,
                maximum_stages=32,minimum_fraction=1/1024,iterations_per_stage=160,tolerance=1e-7):
    """Bounded feasible continuation, in caller coordinates (prefer design units).

    Returns a proposal only. It does not certify global nonoverlap or bone
    clearance and does not rebuild a physical state inside an existing solver.
    Convergence and the 0.2 numerical noninversion floor do not grant admission.
    """
    rest=np.asarray(rest,float);fixed=np.asarray(anchors['nodes'],np.int64);targets=np.asarray(anchors['targets'],float)
    if targets.shape!=rest.shape or not np.isfinite(targets).all():raise ValueError('Invalid target field')
    if not np.any(targets[fixed]-rest[fixed]):
        return rest.copy(),dict(converged=True,fraction=1.,stages=[],identity_exact=True)
    objective=RestMeshObjective(rest,tets,minimum_jacobian=minimum_jacobian)
    free=np.setdiff1d(np.arange(len(rest)),fixed);delta=np.zeros_like(rest);delta[fixed]=targets[fixed]-rest[fixed]
    if len(free):
        factor=splu(objective.K[free][:,free].tocsc())
        delta[free]=factor.solve(-objective.K[free][:,fixed]@delta[fixed])
    x=rest.copy();fraction=0.;increment=initial_fraction;stages=[]
    for _ in range(maximum_stages):
        step=min(increment,1-fraction);trial=x+step*delta
        trial[fixed]=rest[fixed]+(fraction+step)*(targets[fixed]-rest[fixed])
        valid=segment_minimum_jacobian(objective.deformation(x),objective.deformation(trial))>minimum_jacobian
        if valid:trial,status=_relax(objective,trial,fixed,iterations_per_stage,tolerance)
        else:status=dict(converged=False,reason='anchor-increment-inversion')
        stages.append(dict(from_fraction=fraction,increment=step,**status))
        if not status['converged']:
            increment=step*.5
            if increment<minimum_fraction:break
            continue
        x=trial;fraction+=step
        if fraction>=1-1e-14:
            x[fixed]=targets[fixed]
            return x,dict(converged=True,fraction=1.,stages=stages,identity_exact=False)
        increment=min(step*1.5,1-fraction)
    return x,dict(converged=False,fraction=fraction,stages=stages,reason='bounded-authoring-budget')

def regenerate_reference(old_arrays,new_rest,muscles):
    """Rebuild every geometry-dependent FEM record; preserve material constants."""
    old=np.asarray(old_arrays['rest']);x=np.asarray(new_rest);t=np.asarray(old_arrays['nodes'])
    dm=cell_matrices(x,t);old_dm=cell_matrices(old,t);volume=np.linalg.det(dm)/6
    if not np.isfinite(x).all() or volume.min()<=0:raise ValueError('Invalid new rest')
    grad=np.zeros((len(t),4,3));grad[:,1:]=np.linalg.inv(dm);grad[:,0]=-grad[:,1:].sum(1)
    material=np.array(old_arrays['material'],copy=True);material[:,0]=volume
    mass=np.zeros(len(x))
    for a in range(4):np.add.at(mass,t[:,a],1000*volume/4)
    F=dm@np.linalg.inv(old_dm);regions=copy.deepcopy(muscles)
    for r in regions:
        ids=np.asarray(r['tetrahedra']);f=np.asarray(r['fibreDirections'],float)
        f=np.einsum('tij,tj->ti',F[ids],f);length=np.linalg.norm(f,axis=1)
        if not np.isfinite(f).all() or np.any(length<=1e-14):raise ValueError('Invalid transported fibre')
        r['fibreDirections']=(f/length[:,None]).tolist()
    new={k:np.array(v,copy=True) for k,v in old_arrays.items()}
    new.update(rest=x.copy(),gradients=grad,material=material,mass=mass)
    ratio=np.linalg.det(F);condition=np.linalg.cond(dm);baseline_condition=np.linalg.cond(old_dm)
    report=dict(positive_determinants=True,minimum_signed_volume=float(volume.min()),
        relative_jacobian_range=[float(ratio.min()),float(ratio.max())],
        Dm_condition_quantiles=np.quantile(condition,[0,.5,.95,.99,1]).tolist(),
        maximum_Dm_condition_ratio=float((condition/baseline_condition).max()),
        muscle_families_transported=len(regions),mass_density_solver_units=1000.,
        overlap_audit='NOT RUN',bone_clearance_admission='NOT RUN',dense_skin_contact='NOT RUN',
        admission_requires='Independent local-J, conditioning, overlap and contact comparison against baseline; numerical convergence alone is insufficient.',
        accepted_for_mechanics=False)
    return new,regions,report

def retarget_binding(binding,new_head,new_rest,geometry_hash,head_hash):
    """Preserve exact chart identities/weights; replace geometry and provenance."""
    b={k:np.array(v,copy=True) for k,v in binding.items()};ix=b['indices'];w=b['weights']
    b['points']=np.asarray(new_head)[b['vertex_ids']]*float(b['solver_units_per_design_unit'])
    b['rest_offsets']=b['points']-(new_rest[ix]*w[:,:,None]).sum(1)
    b['parent_source_head_sha256']=b['source_head_sha256'].copy()
    b['parent_source_geometry_sha256']=b['source_geometry_sha256'].copy()
    b['source_head_sha256']=np.array(head_hash);b['source_geometry_sha256']=np.array(geometry_hash)
    b['required_gradient_recovery_count']=np.array(len(new_rest),dtype=np.int32)
    b['rest_authoring_note']=np.array('Original topology/material UV correspondence, new explicitly authored reference. Historical patch/oral source hashes are provenance, not new coordinate claims.')
    return b

def load_contract(case,binding_path,baseline_head,candidate_head,selected_head_sha256=None):
    case=Path(case);d=json.loads((case/'domain.json').read_text());arrays=read_arrays(case,d)
    if sha(binding_path)!=d['native_binding_sha256']:raise ValueError('Binding pin differs')
    with np.load(binding_path) as z:b={k:z[k] for k in z.files}
    if sha(baseline_head)!=str(b['source_head_sha256']):raise ValueError('Baseline head pin differs')
    candidate_hash=sha(candidate_head)
    if selected_head_sha256 is not None and candidate_hash!=selected_head_sha256:raise ValueError('Selected artistic head pin differs')
    with np.load(baseline_head) as z:old=z['vertices'];faces=z['triangles'];objects=z['object_ids']
    with np.load(candidate_head) as z:
        new=z['vertices']
        if not np.array_equal(faces,z['triangles']) or not np.array_equal(objects,z['object_ids']):raise ValueError('Native topology/object identities changed')
    rigid=np.unique(np.concatenate([q['fixed_collar_nodes'] for q in d['distributed_origin_attachments']]))
    roots=np.concatenate([q['root_nodes'] for q in d['distributed_origin_attachments']])
    if (len(np.unique(roots))!=len(roots) or not np.isin(roots,rigid).all() or
        not np.array_equal(np.sort(roots),np.sort(d['immutable_bonded_root_nodes']))):
        raise ValueError('Root identities must be unique, declared and included in the rigid collar')
    scale=float(d['coordinate_scale_solver_per_design_unit'])
    if not np.isfinite(scale) or scale<=0 or scale!=float(b['solver_units_per_design_unit']):
        raise ValueError('Case and binding coordinate scales differ')
    anchors=exact_native_anchors(arrays['rest'],b,old,new,rigid,scale)
    # Check the fixed bone equality before proposing any change.
    bone=np.fromfile(case/d['bones']['binary'],dtype='<f8',count=d['bones']['vertex_count']*3).reshape(-1,3)
    for q in d['distributed_origin_attachments']:
        if not np.array_equal(arrays['rest'][q['root_nodes']],bone[q['root_combined_bone_vertex_ids']]):raise ValueError('Baseline finite roots differ from named bone vertices')
    return d,arrays,b,old,new,anchors,candidate_hash

def write_authored_case(case,output,d,old_arrays,new_arrays,regions,binding,new_head,
                        selected_head_sha256,solver_report,quality_report):
    """Persist a NEW, explicitly unaccepted case after bounded authoring succeeds.

    Caller must have validated the selected head pin with load_contract. This
    function never grants mechanics/overlap approval or reuses old solved states.
    """
    case=Path(case);out=Path(output)
    if out.exists():raise FileExistsError('Never overwrite a rest checkpoint')
    if not solver_report.get('converged'):raise ValueError('Unconverged rest authoring')
    rigid=np.unique(np.concatenate([q['fixed_collar_nodes'] for q in d['distributed_origin_attachments']]))
    roots=np.concatenate([q['root_nodes'] for q in d['distributed_origin_attachments']])
    if (len(np.unique(roots))!=len(roots) or not np.isin(roots,rigid).all() or
        not np.array_equal(np.sort(roots),np.sort(d['immutable_bonded_root_nodes']))):
        raise ValueError('Writer requires exact declared roots inside the rigid collar')
    if not np.array_equal(new_arrays['rest'][rigid],old_arrays['rest'][rigid]):raise ValueError('Rigid root/collar coordinates changed')
    bone=np.fromfile(case/d['bones']['binary'],dtype='<f8',count=d['bones']['vertex_count']*3).reshape(-1,3)
    for q in d['distributed_origin_attachments']:
        if not np.array_equal(new_arrays['rest'][q['root_nodes']],bone[q['root_combined_bone_vertex_ids']]):
            raise ValueError('Writer refuses a root differing from its named bone vertex')
    out.mkdir(parents=True);contract=write_arrays(out,new_arrays)
    # Copy rather than mutate or hard-link an authoritative source file.
    (out/'bones.bin').write_bytes((case/d['bones']['binary']).read_bytes())
    np.savez_compressed(out/'rest_geometry.npz',vertices=new_arrays['rest']/float(d['coordinate_scale_solver_per_design_unit']),
        tetrahedra=new_arrays['nodes'],boundary_faces=np.asarray(d['boundary_triangles']))
    geometry_hash=sha(out/'rest_geometry.npz')
    b=retarget_binding(binding,new_head,new_arrays['rest'],geometry_hash,selected_head_sha256)
    np.savez_compressed(out/'native_surface_binding.npz',**b)
    result=copy.deepcopy(d);result.update(arrays=contract,muscles=regions,
        source_geometry_sha256=geometry_hash,native_binding_sha256=sha(out/'native_surface_binding.npz'),
        parent_case_domain_sha256=sha(case/'domain.json'),
        status='UNACCEPTED authored rest candidate; new overlap/containment/bone admission, transfer checks and undriven evaluation required')
    result['rest_authoring']={'selected_head_sha256':selected_head_sha256,
        'baseline_binding_sha256':d['native_binding_sha256'],'source_script_sha256':sha(__file__),
        'rigid_root_collar_preserved':True,'old_simulation_cut_ids_repin_to_new_rest':True,
        'reference_reset':'Explicit new artistic reference, not a physical trajectory or preservation of pre-stress.'}
    result['limits']=list(result.get('limits',[]))+['Prior state, trajectory and geometry acceptance does not transfer to this authored rest case.']
    (out/'domain.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    report={**quality_report,'solver':solver_report,'root_collar_bit_exact':True,
        'domain_sha256':sha(out/'domain.json'),'arrays_sha256':sha(out/'arrays.bin'),
        'bones_sha256':sha(out/'bones.bin'),'binding_sha256':result['native_binding_sha256'],
        'native_rest_baseline_sha256':selected_head_sha256,'accepted_for_mechanics':False}
    (out/'rest_authoring_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def author_selected_case(case,binding_path,baseline_head,candidate_head,selected_head_sha256,
                         output=None,solve_options=None):
    """Explicit full-work entry point, never invoked by the inspection CLI.

    The caller must select and approve an artistic head BEFORE scheduling this
    bounded job. Merely discovering a candidate does not authorize this call.
    Returns proposed data; output, when requested, is a new unaccepted checkpoint.
    """
    if not isinstance(selected_head_sha256,str) or len(selected_head_sha256)!=64:
        raise ValueError('An explicit selected artistic SHA256 is required')
    d,old,b,base,new,anchors,pin=load_contract(case,binding_path,baseline_head,candidate_head,selected_head_sha256)
    scale=float(d['coordinate_scale_solver_per_design_unit'])
    design_anchors={**anchors,'targets':anchors['targets']/scale}
    x,solve=extend_rest(old['rest']/scale,old['nodes'],design_anchors,**(solve_options or {}))
    if not solve['converged']:raise RuntimeError('Bounded rest authoring did not converge: '+json.dumps(solve))
    new_rest=x*scale
    # Preserve exact binary Dirichlet values after design/solver-unit round trips.
    new_rest[anchors['nodes']]=anchors['targets'][anchors['nodes']]
    arrays,regions,quality=regenerate_reference(old,new_rest,d['muscles'])
    minimum=(solve_options or {}).get('minimum_jacobian',.2)
    if quality['relative_jacobian_range'][0]<=minimum:raise ValueError('Final unit conversion failed relative-J gate')
    quality.update(surface_anchor_count=len(anchors['surface_nodes']),rigid_root_collar_count=len(anchors['rigid_nodes']),
        surface_anchor_maximum_error=float(abs(new_rest[anchors['surface_nodes']]-anchors['targets'][anchors['surface_nodes']]).max(initial=0)),
        root_collar_bit_exact=bool(np.array_equal(new_rest[anchors['rigid_nodes']],old['rest'][anchors['rigid_nodes']])))
    if output is not None:
        quality=write_authored_case(case,output,d,old,arrays,regions,b,new,pin,solve,quality)
    return arrays,regions,quality,solve

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Read-only exact-anchor inspection; does NOT run the full authoring solve.')
    p.add_argument('--case',type=Path,required=True);p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--baseline-head',type=Path,required=True);p.add_argument('--candidate-head',type=Path,required=True)
    args=p.parse_args();d,a,b,old,new,anchors,pin=load_contract(args.case,args.binding,args.baseline_head,args.candidate_head)
    moved=np.linalg.norm(anchors['targets'][anchors['surface_nodes']]-a['rest'][anchors['surface_nodes']],axis=1)>1e-13
    print(json.dumps(dict(candidate_head_sha256=pin,surface_anchors=len(anchors['surface_nodes']),
        rigid_root_collar_anchors=len(anchors['rigid_nodes']),moved_surface_anchors=int(moved.sum()),
        full_solve_run=False,accepted_for_mechanics=False),indent=2))
