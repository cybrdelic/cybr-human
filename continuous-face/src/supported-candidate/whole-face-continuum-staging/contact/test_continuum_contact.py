"""Bounded manufactured qualification only. No actual-model solve is launched."""
from pathlib import Path
import json,time,sys
from types import SimpleNamespace
import numpy as np
from scipy.sparse import eye,csr_matrix
from continuum_contact import (ContinuumContactGuard,boundary_carrier,topology_report,
    ClosedTriangleOwner,UnsignedTriangleSheetOwner,create_guard)
from volume_guard import certify_positive_volume
from carrier_partition import CarrierLocalPartition,CarrierPartitionedSelfContact,exact_unit_sum,unresolved_adjacency_pairs
from unified_moving_surface_contact import NonsharedSelfContact,material_ids,adjacency_pairs
from conservative_broadphase import candidates,exhaustive
from implicit_dynamics import ImplicitDynamics,MissingGeometryCertification

HERE=Path(__file__).resolve().parent
BASE=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
F=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
EVIDENCE={k:True for k in ('source_geometry_qualified','native_rigid_inventory_complete',
    'cavity_inventory_complete','boundary_is_physical_material_boundary',
    'rigid_source_self_intersections_checked','reference_boundary_embedding_certified')}

def solid_at(offsets,scales=None):
    if scales is None:scales=np.ones(len(offsets))
    X=np.concatenate([BASE*s+o for o,s in zip(offsets,scales)])
    T=np.arange(len(X)).reshape(-1,4)
    dm=np.stack([X[T[:,j]]-X[T[:,0]] for j in (1,2,3)],axis=2)
    S=SimpleNamespace(X=X,T=T,inv=np.linalg.inv(dm),mm_per_L0=1.,provenance={'kind':'synthetic finite tetra fixture'})
    S.deformation=lambda q:np.stack([q[T[:,j]]-q[T[:,0]] for j in (1,2,3)],axis=2)@S.inv
    return S

def build(solid,**kwargs):
    faces=np.concatenate([F+4*i for i in range(len(solid.T))])
    return ContinuumContactGuard(solid,eye(len(solid.X),format='csr'),faces,
        boundary_class=np.zeros(len(faces),int),initial_evidence=EVIDENCE,
        attachment_evidence={'finite_material_attachment_qualified':True,
            'interpretation':'synthetic fixture supports only; no anatomy attachment claim'},
        fixture_only=True,**kwargs)


def main():
    started=time.perf_counter();rows=[]
    def test(name,fn):
        t=time.perf_counter();result=fn();rows.append(dict(name=name,passed=True,seconds=time.perf_counter()-t,**(result or {})));print(name,'PASS',flush=True)

    def mapping():
        s=solid_at([np.zeros(3)])
        C,f,r=boundary_carrier(s,s.X,F,np.zeros(4,int),np.eye(4))
        assert np.array_equal(C@s.X,s.X)
        try:boundary_carrier(s,s.X+.01,F,np.zeros(4,int),np.eye(4))
        except ValueError:pass
        else:raise AssertionError('mismapped boundary accepted')
        assert topology_report(F,4)['certified']
        assert not topology_report(F[:-1],4)['certified']
        assert not topology_report(np.r_[F,F[:1]],4)['certified']
        return r
    test('carrier_map_and_closed_topology',mapping)

    def no_contact():
        s=solid_at([np.zeros(3),np.array([4.,0,0])]);g=build(s)
        q=s.X+np.array([.2,.1,.3]);path=g.path(s.X,q);assert path['whole_segment_certified']
        e,grad=g.energy(q);assert e==0 and np.count_nonzero(grad)==0 and g.tangent(q).nnz==0
        assert g.safe_step_bound(s.X,q-s.X)==1
        assert len(g.shared.pairs)==0
        return dict(path=path,partition=g.partition.report,energy=e)
    test('no_contact_translation_with_local_affine_proof',no_contact)

    def crossing():
        s=solid_at([np.zeros(3),np.array([3.,0,0])]);g=build(s)
        q=s.X.copy();q[:4,0]+=3;q[4:,0]-=3
        assert g.endpoint(s.X)['feasible'] and g.endpoint(q)['feasible']
        p=g.path(s.X,q);assert not p['whole_segment_certified']
        bound=g.safe_step_bound(s.X,q-s.X);assert 0<bound<1
        assert g.path(s.X,s.X+bound*(q-s.X))['whole_segment_certified']
        return dict(full_crossing_rejected=True,safe_step=bound)
    test('clear_endpoint_midpath_self_crossing',crossing)

    def tangency():
        s=solid_at([np.zeros(3),np.array([2.,0,0])]);g=build(s)
        q=s.X.copy();q[4:,0]-=1
        assert not g.endpoint(q)['feasible'] and not g.path(s.X,q)['whole_segment_certified']
        return dict(tangent_endpoint_rejected=True)
    test('zero_clearance_tangency',tangency)

    def bone_crossing():
        s=solid_at([np.zeros(3)])
        owner=ClosedTriangleOwner(BASE+np.array([3.,0,0]),F,name='native synthetic rigid')
        g=build(s,rigid_owners=[owner]);q=s.X+np.array([6.,0,0])
        assert g.endpoint(s.X)['feasible'] and g.endpoint(q)['feasible']
        assert not g.path(s.X,q)['whole_segment_certified']
        assert g.safe_step_bound(s.X,q-s.X)<1
        return dict(native_finite_crossing_rejected=True,bone_exclusions=g.report['bone_pair_exclusions'])
    test('finite_rigid_clear_endpoint_crossing',bone_crossing)

    def unsigned_sheet():
        s=solid_at([np.zeros(3)])
        V=np.array([[2.,-3.,-3.],[2.,3.,-3.],[2.,0.,3.]])
        owner=UnsignedTriangleSheetOwner(V,np.array([[0,1,2]]),name='literal open source sheet')
        try:build(s,rigid_owners=[owner])
        except ValueError:pass
        else:raise AssertionError('unqualified open side accepted')
        owner.initial_side_qualified=True
        g=build(s,rigid_owners=[owner]);q=s.X+np.array([4.,0,0])
        assert g.endpoint(s.X)['feasible'] and g.endpoint(q)['feasible']
        assert not g.path(s.X,q)['whole_segment_certified']
        return dict(unknown_initial_side_rejected=True,literal_finite_sheet_crossing_rejected=True,
            inside_sign_claimed=False,limitation='finite two-sided source sheet only; no invented watertight solid')
    test('explicit_open_native_sheet_policy',unsigned_sheet)

    def rigid_union():
        s=solid_at([np.zeros(3)])
        owners=[ClosedTriangleOwner(BASE+np.array([3.+offset,0,0]),F,name='fixed union component '+str(i))
                for i,offset in enumerate((0.,.1))]
        g=build(s,rigid_owners=owners)
        assert g.endpoint(s.X)['feasible'] and g.path(s.X,s.X+np.array([.1,0,0]))['whole_segment_certified']
        return dict(overlapping_static_owners_allowed=True,rigid_rigid_disjointness_required=False)
    test('overlapping_fixed_rigid_union_not_a_blocker',rigid_union)



    def containment():
        s=solid_at([np.array([.1,.1,.1])],scales=[.1])
        g=build(s,rigid_owners=[ClosedTriangleOwner(BASE,F,name='containing bone')])
        assert not g.endpoint(s.X)['feasible']
        s=solid_at([np.zeros(3)],scales=[10.])
        g=build(s,rigid_owners=[ClosedTriangleOwner(BASE+1,F,name='enclosed bone')])
        assert not g.endpoint(s.X)['feasible']
        return dict(soft_inside_bone_rejected=True,bone_inside_soft_rejected=True)
    test('two_way_closed_solid_containment',containment)

    def derivatives():
        s=solid_at([np.zeros(3),np.array([1.01,0,0])]);g=build(s,clearance_mm=.05)
        q=s.X.copy();e,gradient=g.energy(q);assert e>0
        rng=np.random.default_rng(53);errors=[]
        for _ in range(5):
            d=rng.normal(size=q.shape);eps=1e-8
            fd=(g.energy(q+eps*d)[0]-g.energy(q-eps*d)[0])/(2*eps)
            exact=float(np.sum(gradient*d));errors.append(abs(fd-exact)/max(1e-10,abs(exact)))
        assert max(errors)<5e-6,max(errors)
        K=g.tangent(q);assert np.max(abs((K-K.T).data),initial=0)<1e-12
        eig=np.linalg.eigvalsh(K.toarray());assert eig.min()>-1e-12
        return dict(active_energy_N_mm=e,max_gradient_relative_error=max(errors),minimum_GN_eigenvalue=float(eig.min()),contact_tangent='PSD normal GN, not exact Hessian')
    test('finite_contact_force_and_PSD_tangent',derivatives)

    def volume():
        s=solid_at([np.zeros(3)]);q=s.X@np.diag([-2.,-.5,1.])
        assert np.linalg.det(s.deformation(s.X))[0]>0 and np.linalg.det(s.deformation(q))[0]>0
        proof=certify_positive_volume(s,s.X,q);assert not proof['certified']
        r=s.X@np.diag([1.1,.9,1.02]);ok=certify_positive_volume(s,s.X,r);assert ok['certified']
        return dict(clear_endpoint_volume_collapse_rejected=True,rejected_proof=proof,accepted_proof=ok)
    test('whole_segment_positive_volume_not_endpoint_sampling',volume)

    def face_neighbor():
        s=solid_at([np.zeros(3)])
        s.X=np.r_[s.X,[[0.,0.,-1.]]];s.T=np.array([[0,1,2,3],[0,2,1,4]])
        s.inv=np.linalg.inv(np.stack([s.X[s.T[:,j]]-s.X[s.T[:,0]] for j in (1,2,3)],axis=2))
        faces=np.array([[1,2,3],[1,2,4]])
        p=CarrierLocalPartition(s,eye(5,format='csr'),faces,reference_embedding_certified=True)
        assert p.certified_pairs(np.array([[0,1]]))[0]
        # A non-exact partition never gets an algebraic exemption.
        C=eye(5,format='lil');C[0,0]=np.nextafter(1.,0.);C=C.tocsr()
        assert not exact_unit_sum([np.nextafter(1.,0.)])
        p2=CarrierLocalPartition(s,C,np.array([[0,1,3]]),reference_embedding_certified=True)
        assert p2.parents[0]==-1
        return p.report
    test('complete_face_parent_certificate_and_exact_weights',face_neighbor)

    def broadphase():
        rng=np.random.default_rng(834);a=rng.uniform(-3,3,(400,3));b=rng.uniform(-3,3,(350,3))
        ar=rng.uniform(.001,.2,(400,3));br=rng.uniform(.001,.2,(350,3))
        p=candidates(a-ar,a+ar,b-br,b+br,.02);ref=exhaustive(a-ar,a+ar,b-br,b+br,.02)
        assert np.array_equal(p,ref)
        # Bounded large sparse workload, not a facial model or all-pairs campaign.
        side=60;x,y=np.meshgrid(np.arange(side),np.arange(side),indexing='ij')
        lo=np.c_[x.ravel()*2,y.ravel()*2,np.zeros(side*side)];hi=lo+.2
        t=time.perf_counter();large=candidates(lo,hi,lo,hi,.02);elapsed=time.perf_counter()-t
        assert len(large)==len(lo)
        return dict(random_exact_AABB_candidates=len(p),random_allpairs=400*350,
            sparse_boxes=len(lo),sparse_candidates=len(large),hypothetical_allpairs=len(lo)**2,
            candidate_fraction=len(large)/len(lo)**2,wall_seconds=elapsed,method='conservative radius-bucket cKDTree plus exact outward AABB filter')
    test('sparse_broadphase_exact_equivalence_and_cost',broadphase)

    def hierarchy():
        rng=np.random.default_rng(123)
        offsets=np.array([[x*1.5,y*1.5,0.] for x in range(4) for y in range(4)])
        s=solid_at(offsets);faces=np.concatenate([F+4*i for i in range(len(s.T))])
        C=eye(len(s.X),format='csr');ids=material_ids(C)
        partition=CarrierLocalPartition(s,C,faces,reference_embedding_certified=True)
        accelerated=CarrierPartitionedSelfContact(C,faces,ids,partition,clearance=.02)
        original=NonsharedSelfContact(C,faces,ids,clearance=.02)
        expected_shared=adjacency_pairs(faces,ids)
        expected_shared=expected_shared[~partition.certified_pairs(expected_shared)]
        actual_shared,_=unresolved_adjacency_pairs(faces,ids,partition,chunk_size=17)
        assert np.array_equal(expected_shared,actual_shared)
        for _ in range(3):
            q=s.X+rng.normal(size=s.X.shape)*.05;d=rng.normal(size=s.X.shape)*.4
            for direction in (None,d):
                expected=original.candidates(q,direction)
                expected=expected[~partition.certified_pairs(expected)]
                actual=accelerated.candidates(q,direction)
                assert np.array_equal(actual,expected),(actual.shape,expected.shape)
        grid=np.array([[x*2.,y*2.,z*2.] for x in range(10) for y in range(10) for z in range(10)])
        s=solid_at(grid);faces=np.concatenate([F+4*i for i in range(len(s.T))]);C=eye(len(s.X),format='csr')
        t=time.perf_counter();partition=CarrierLocalPartition(s,C,faces,reference_embedding_certified=True)
        term=CarrierPartitionedSelfContact(C,faces,material_ids(C),partition,clearance=.02)
        build_time=time.perf_counter()-t;t=time.perf_counter();pairs=term.candidates(s.X);query_time=time.perf_counter()-t
        assert not len(pairs)
        return dict(exact_static_and_swept_equivalence_cases=6,carrier_tets=len(s.T),boundary_faces=len(faces),
            build_seconds=build_time,query_seconds=query_time,**term.last_candidate_report)
    test('two_level_parent_and_leaf_broadphase',hierarchy)


    def fail_closed():
        s=solid_at([np.zeros(3)]);s.provenance={'domain_path':str(HERE.parent/'geometry/whole-face-continuum-final.npz')}
        guard=create_guard(s,None)
        assert not any(guard.certification.values())
        try:ImplicitDynamics(s,guard).require_guards()
        except MissingGeometryCertification:pass
        else:raise AssertionError('unqualified voxel baseline accepted')
        return guard.report
    test('voxel_baseline_fail_closed',fail_closed)

    def canonical_fixture():
        import tempfile
        from canonicalize_boundary_map import canonicalize
        with tempfile.TemporaryDirectory(prefix='continuum-map-') as folder:
            folder=Path(folder);s=solid_at([np.zeros(3)])
            W=np.eye(4);W[0,0]=np.nextafter(1.,0.);W[0,1]=-1e-16
            source=folder/'raw.npz';output=folder/'derived.npz'
            np.savez_compressed(source,nodes=s.X,tets=s.T,vertices=s.X,triangles=F,
                exterior_tet=np.zeros(4,int),exterior_barycentric=W,boundary_parent_tet=np.zeros(4,int),boundary_class=np.zeros(4,int))
            r=canonicalize(source,output)
            assert r['all_vertex_partitions_exact_binary_rational'] and r['all_facets_exactly_in_declared_parent']
            assert r['maximum_reference_shift_L0']<1e-10 and r['source_file_unchanged']
            assert not r['motion_certified']
            return dict(exact_partition=True,source_preserved=True,no_motion_certificate=True,
                max_shift=r['maximum_reference_shift_L0'])
    test('disclosed_dyadic_representation_repair',canonical_fixture)

    def global_injectivity():
        from itertools import permutations
        from grid_injectivity import RegularGridInjectivity
        X=np.array([[x,y,z] for x in (0.,1.,2.) for y in (0.,1.) for z in (0.,1.)])
        lookup={tuple(x):i for i,x in enumerate(X)};T=[]
        for x in (0,1):
            for perm in permutations(range(3)):
                chain=[np.array([x,0,0])]
                for axis in perm:
                    point=chain[-1].copy();point[axis]+=1;chain.append(point)
                t=[lookup[tuple(p)] for p in chain]
                dm=np.stack([X[t[j]]-X[t[0]] for j in (1,2,3)],axis=1)
                if np.linalg.det(dm)<0:t[1],t[2]=t[2],t[1]
                T.append(t)
        T=np.array(T);s=SimpleNamespace(X=X,T=T,inv=np.linalg.inv(np.stack([X[T[:,j]]-X[T[:,0]] for j in (1,2,3)],axis=2)),mm_per_L0=1.,provenance={'kind':'synthetic regular grid'})
        s.deformation=lambda q:np.stack([q[T[:,j]]-q[T[:,0]] for j in (1,2,3)],axis=2)@s.inv
        proof=RegularGridInjectivity(s,1.)
        q=X@np.diag([1.1,.9,1.05]);report=proof.path(X,q);assert report['certified']
        actual=np.linalg.norm(np.diag([.1,-.1,.05]),'fro')
        assert report['end']['maximum_gradient_Frobenius_upper']>=actual
        assert not proof.path(X,X@np.diag([-1.,1.,1.]))['certified']
        assert proof.direction_path(X,q-X)['certified']
        # Floating origin+h*i equality is insufficient: this translated grid
        # rounds its cell widths and must not get the exact-grid shortcut.
        try:RegularGridInjectivity(SimpleNamespace(X=X+.1,T=T),1.)
        except ValueError:pass
        else:raise AssertionError('only approximately affine reference grid accepted')

        # Incomplete active grid leaves proof-only background nodes at zero.
        use=np.unique(T[:1]);renumber={v:i for i,v in enumerate(use)}
        small=SimpleNamespace(X=X[use],T=np.array([[renumber[v] for v in T[0]]]))
        extension=RegularGridInjectivity(small,1.)
        assert extension.report['proof_only_zero_displacement_nodes']>0
        assert extension.bound(small.X+.001)['certified']
        # Reject a standard simplex that is not in this Freudenthal grid.
        try:RegularGridInjectivity(solid_at([np.zeros(3)]),1.)
        except ValueError:pass
        else:raise AssertionError('incompatible grid simplex accepted')
        allfaces=np.concatenate([T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]])
        _,inverse,counts=np.unique(np.sort(allfaces,axis=1),axis=0,return_inverse=True,return_counts=True)
        faces=allfaces[counts[inverse]==1]
        guard=ContinuumContactGuard(s,eye(len(X),format='csr'),faces,boundary_class=np.zeros(len(faces),int),initial_evidence=EVIDENCE,
            attachment_evidence={'finite_material_attachment_qualified':True},fixture_only=True,grid_spacing_mm=1.)
        path=guard.path(X,q);assert path['whole_segment_certified']
        assert path['surfaces'][0]['name']=='global_carrier_injectivity'
        # A geometric shortcut does not replace finite self barrier energy.
        e,g=guard.energy(q);direct,gradient=guard.nonshared(q)
        assert e==guard.stiffness*direct and np.array_equal(g,guard.stiffness*gradient)
        return dict(path=report,background=extension.report,finite_energy_unchanged=True,
            actual_facial_motion_claimed=False)
    test('regular_grid_global_injectivity_and_fallback',global_injectivity)

    def ragged_equivalence():
        from ragged_group_pairs import candidates as ragged
        from conservative_broadphase import candidates as spatial
        rng=np.random.default_rng(6271);cases=[]
        for shift in (0.,1.e12):
            for h in (0.,.02,1.):
                lo=rng.uniform(-2,2,(63,3))+shift
                hi=lo+rng.uniform(0,2,(63,3))
                labels=rng.integers(0,9,len(lo));groups=np.array([(a,b) for a in range(9) for b in range(a+1,9) if (a+b)%3],int)
                ids=rng.integers(0,500,(len(lo),3))
                ref=spatial(lo,hi,lo,hi,h);ref=ref[ref[:,0]<ref[:,1]]
                keys=np.minimum(labels[ref[:,0]],labels[ref[:,1]])*9+np.maximum(labels[ref[:,0]],labels[ref[:,1]])
                ref=ref[np.isin(keys,groups[:,0]*9+groups[:,1])]
                shared=(ids[ref[:,0],:,None]==ids[ref[:,1],None,:]).any((1,2));ref=ref[~shared]
                for chunk in (1,17,257):
                    actual,report=ragged(lo,hi,labels,groups,ids,h,batch_size=chunk,return_report=True)
                    assert np.array_equal(actual,ref)
                cases.append(dict(translation=shift,clearance=h,pairs=len(ref)))
        # Group order differs from fine face order. The exact original AABB
        # predicate must still be evaluated with ascending fine face IDs.
        lo=np.array([[1.,0,0],[0.,0,0]]);hi=lo.copy();labels=np.array([1,0]);ids=np.array([[0,1,2],[3,4,5]])
        for h in (np.nextafter(1.,0.),1.,np.nextafter(1.,np.inf)):
            ref=spatial(lo,hi,lo,hi,h);ref=ref[ref[:,0]<ref[:,1]]
            assert np.array_equal(ragged(lo,hi,labels,[[0,1]],ids,h,batch_size=1),ref)
        return dict(random_translated_and_clearance_cases=cases,bounded_batch_sizes=[1,17,257],
            reversed_group_face_order_and_rounding_boundary=True,exact_candidate_identity=True)
    test('ragged_parent_cartesian_exact_candidate_identity',ragged_equivalence)




    report=dict(status='synthetic_adapter_qualification_passed',tests=rows,tests_passed=len(rows),
        wall_seconds=time.perf_counter()-started,actual_facial_frames=0,
        actual_source_geometry_certified=False,actual_source_attachment_certified=False,
        scientific_validation=False,scope='bounded finite fixtures and sparse candidate benchmark only')
    (HERE/'synthetic-qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2))

if __name__=='__main__':main()
