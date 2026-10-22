"""One bounded, construction-only self-force diagnostic; never a source certificate.

Uses the exact production finite pair law and sparse carrier Jacobian in chunks.
The existing local-parent exclusion is enabled CONDITIONALLY on the marching-tet
reference construction, not asserted as an independently proved embedding of the
separately dyadic-rounded reference. Native contacts are deliberately not mixed
into this self-force diagnostic. No solver, time integration, or source edits.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,time,signal
import numpy as np
from scipy.sparse import csr_matrix
from continuum_contact import sha
from carrier_partition import CarrierLocalPartition,CarrierPartitionedSelfContact
from unified_moving_surface_contact import material_ids
from triangle_interface_contact import triangle_geometry,activated_log_barrier
from grid_injectivity import RegularGridInjectivity


def diagnose(source,boundary,output,*,seconds=145,max_candidates=2_000_000,chunk=4096):
    source,boundary,output=map(Path,(source,boundary,output))
    if output.exists():raise ValueError('new diagnostic output path required')
    start=time.perf_counter()
    report=dict(status='running_construction_only_diagnostic',actual_facial_frames=0,
        accepted_motion=False,source_state_certified=False,reference_embedding_certified=False,
        native_contact_checked=False,native_reference_known_failed=True,
        conditional_assumption='local-parent exclusions assume the source conforming marching-tet embedding; dyadic rounded reference embedding is not independently certified',
        source_path=str(source.resolve()),source_sha256=sha(source),
        boundary_path=str(boundary.resolve()),boundary_sha256=sha(boundary),
        numerical_model=dict(clearance_mm=.02,stiffness_N_per_mm=.001,unit_facet_weights=True,
            local_same_material_adjacency_barrier=0.,scientific_calibration=False),
        time_budget_seconds=seconds,candidate_budget=max_candidates)
    def checkpoint(stage):
        report.update(stage=stage,wall_seconds=time.perf_counter()-start)
        output.write_text(json.dumps(report,indent=2)+'\n');print(stage,report['wall_seconds'],flush=True)
    def expired(*args):raise TimeoutError('bounded diagnostic wall-time budget exhausted')
    signal.signal(signal.SIGALRM,expired);signal.alarm(seconds)
    try:
        checkpoint('loading')
        with np.load(source,allow_pickle=False) as z:X,T=[z[k].copy() for k in ('nodes','tets')]
        with np.load(boundary,allow_pickle=False) as z:
            V,F,P,W,K=[z[k].copy() for k in ('vertices','triangles','exterior_tet','exterior_barycentric','boundary_class')]
            assert str(z['source_domain_sha256'])==report['source_sha256']
        C=csr_matrix((W.ravel(),(np.repeat(np.arange(len(V)),4),T[P].ravel())),shape=(len(V),len(X)))
        C.eliminate_zeros();C.sort_indices();s=SimpleNamespace(X=X,T=T)
        assert np.array_equal(C@X,V)
        report.update(physical_nodes=len(X),carrier_tets=len(T),boundary_vertices=len(V),boundary_triangles=len(F))
        t=time.perf_counter();grid=RegularGridInjectivity(s,6.)
        report['reference_grid_proof']=dict(construction=grid.report,bound=grid.bound(X),seconds=time.perf_counter()-t)
        # Diagnostic-only conditional use. This result is NEVER written into a
        # guard manifest or claimed as independently certified source embedding.
        t=time.perf_counter();part=CarrierLocalPartition(s,C,F,reference_embedding_certified=True)
        report['conditional_local_partition']=dict(**part.report,seconds=time.perf_counter()-t)
        ids=material_ids(C)
        term=CarrierPartitionedSelfContact(C,F,ids,part,clearance=.02)
        checkpoint('conditional partition ready')
        t=time.perf_counter();pairs=term.candidates(X)
        report['candidate_broadphase']=dict(**term.last_candidate_report,seconds=time.perf_counter()-t)
        checkpoint('sparse finite candidates ready')
        if len(pairs)>max_candidates:raise RuntimeError('candidate budget exceeded; no truncated total claimed')
        V=C@X;g=np.zeros_like(X);energy=0.;active=0;bad=0;minimum=np.inf;hot=[];bad_examples=[]
        for lo in range(0,len(pairs),chunk):
            p=pairs[lo:lo+chunk];geo=triangle_geometry(V[F[p[:,0]]],V[F[p[:,1]]])
            ok=~geo['intersect'];bad+=int(np.sum(~ok));minimum=min(minimum,float(geo['gap'].min(initial=np.inf)))
            for index in np.flatnonzero(~ok)[:max(0,20-len(bad_examples))]:
                bad_examples.append(dict(faces=p[index].tolist(),gap_mm=float(geo['gap'][index]),
                    parent_tets=part.parents[p[index]].tolist(),vertices_a=V[F[p[index,0]]].tolist(),vertices_b=V[F[p[index,1]]].tolist()))
            choose=ok&(geo['gap']<.02);active+=int(np.sum(choose))
            if not np.any(choose):continue
            ap=p[choose];ag={key:value[choose] for key,value in geo.items()}
            b,db,_=activated_log_barrier(ag['gap'],.02)
            state=SimpleNamespace(pairs=ap,geometry=ag);jac=term.jacobian(state)
            g+=.001*np.asarray(jac.T@db).reshape(X.shape);energy+=.001*float(b.sum())
            for index in np.argsort(b)[-min(20,len(b)):]:
                a,c=ap[index];pa,pc=part.parents[[a,c]]
                hot.append(dict(faces=[int(a),int(c)],gap_mm=float(ag['gap'][index]),energy_N_mm=float(.001*b[index]),
                    parent_tets=[int(pa),int(pc)],shared_parent_nodes=int(len(np.intersect1d(T[pa],T[pc]))),
                    boundary_classes=K[[a,c]].tolist(),closest_a_mm=(ag['bary_a'][index]@V[F[a]]).tolist(),
                    closest_b_mm=(ag['bary_b'][index]@V[F[c]]).tolist()))
            hot=sorted(hot,key=lambda row:row['energy_N_mm'],reverse=True)[:20]
        norms=np.linalg.norm(g,axis=1);top=np.argsort(norms)[-20:][::-1]
        report.update(status='construction_only_initial_self_force_measured' if not bad else 'construction_only_self_geometry_failed',
            finite_candidates_evaluated=len(pairs),active_separated_pairs=active,
            intersecting_or_unresolved_pairs=bad,minimum_candidate_gap_mm=None if not np.isfinite(minimum) else minimum,
            self_energy_N_mm=energy if not bad else None,self_gradient_N_l2=float(np.linalg.norm(g)) if not bad else None,
            self_max_node_force_N=float(norms.max(initial=0)) if not bad else None,
            feasible_pairs_only_energy_N_mm=energy,feasible_pairs_only_max_node_force_N=float(norms.max(initial=0)),
            hot_pairs=hot,failed_pair_examples=bad_examples,
            top_force_nodes=[dict(node=int(i),reference_mm=X[i].tolist(),force_N=(-g[i]).tolist(),norm_N=float(norms[i])) for i in top],
            diagnostic_force_sign='negative energy gradient',
            unchanged_finite_triangle_law=True,nonlocal_clearance_not_weakened=True,
            explanation='Shared topological and exact same/common-face parent material pairs have zero artificial barrier. All remaining nonshared candidate pairs use unchanged finite triangle distance, separation checks and activated log barrier. No native forces are included.')
        np.savez_compressed(output.with_suffix('.npz'),self_gradient_N=g,source_domain_sha256=np.array(report['source_sha256']),accepted_motion=np.array(False))
        checkpoint('done')
    except (TimeoutError,RuntimeError,ValueError,MemoryError) as exc:
        report.update(status='bounded_diagnostic_incomplete',reason=str(exc),complete_energy_or_force_claimed=False)
        checkpoint('stopped without truncation claim')
    finally:signal.alarm(0)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('boundary');p.add_argument('output');args=p.parse_args()
    result=diagnose(args.source,args.boundary,args.output)
    print(json.dumps({k:result.get(k) for k in ['status','wall_seconds','active_separated_pairs','intersecting_or_unresolved_pairs','self_energy_N_mm','self_max_node_force_N','reason']},indent=2))
