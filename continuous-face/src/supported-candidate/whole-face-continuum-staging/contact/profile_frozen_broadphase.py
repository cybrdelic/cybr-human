"""One instrumented baseline; cache identities so later tests never repeat it."""
from pathlib import Path
from types import SimpleNamespace
import cProfile,pstats,json,time,signal,hashlib
import numpy as np
from scipy.sparse import csr_matrix
from continuum_contact import sha
import carrier_partition as module
from carrier_partition import CarrierLocalPartition,CarrierPartitionedSelfContact
from unified_moving_surface_contact import material_ids
from moving_triangle_contact import _map_interval
from ragged_group_pairs import candidates as ragged


def main():
    root=Path(__file__).resolve().parent;source=root.parent/'geometry/cut-pilot/repaired-v1/cut-continuum.npz';boundary=root/'repaired-v1-dyadic-boundary.npz'
    output=root/'repaired-v1-broadphase-profile.json';cache=root/'repaired-v1-broadphase-frozen.npz'
    if output.exists() or cache.exists():raise ValueError('Do not repeat the instrumented full baseline; use the saved frozen input')
    signal.alarm(180);started=time.perf_counter()
    report=dict(status='running_bounded_frozen_candidate_profile',source_sha256=sha(source),boundary_sha256=sha(boundary),accepted_motion=False,actual_facial_frames=0,force_or_geometry_law_changed=False)
    def checkpoint(stage):
        report.update(stage=stage,wall_seconds=time.perf_counter()-started);output.write_text(json.dumps(report,indent=2)+'\n');print(stage,report['wall_seconds'],flush=True)
    checkpoint('loading exact frozen source')
    with np.load(source) as z:X,T=z['nodes'],z['tets']
    with np.load(boundary) as z:V,F,P,W=[z[k] for k in ('vertices','triangles','exterior_tet','exterior_barycentric')]
    C=csr_matrix((W.ravel(),(np.repeat(np.arange(len(V)),4),T[P].ravel())),shape=(len(V),len(X)));C.eliminate_zeros();C.sort_indices()
    partition=CarrierLocalPartition(SimpleNamespace(X=X,T=T),C,F,reference_embedding_certified=True)
    term=CarrierPartitionedSelfContact(C,F,material_ids(C),partition,clearance=.02,leaf_strategy='spatial')
    original=module.spatial_candidates;stage_calls=[];coarse_raw=[]
    def measured(*args):
        t=time.perf_counter();p=original(*args);stage_calls.append(dict(boxes_a=len(args[0]),boxes_b=len(args[2]),pairs=len(p),seconds=time.perf_counter()-t))
        if len(stage_calls)==1:coarse_raw.append(p.copy())
        return p
    module.spatial_candidates=measured
    profiler=cProfile.Profile();checkpoint('one instrumented baseline begins')
    t=time.perf_counter();profiler.enable();baseline=term.candidates(X);profiler.disable();elapsed=time.perf_counter()-t
    module.spatial_candidates=original
    stats=pstats.Stats(profiler);top=sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:20]
    report['baseline']=dict(seconds=elapsed,stages=stage_calls,report=term.last_candidate_report,pairs=len(baseline),
        pair_sha256=hashlib.sha256(np.asarray(baseline,dtype='<i8').tobytes()).hexdigest(),
        profile=[dict(file=key[0],line=key[1],function=key[2],calls=data[1],self_seconds=data[2],cumulative_seconds=data[3]) for key,data in top])
    low,high=_map_interval(C,X);lo,hi=low[F].min(1),high[F].max(1)
    groups=coarse_raw[0];groups=groups[groups[:,0]<groups[:,1]]
    a,b=term.group_parent[groups[:,0]],term.group_parent[groups[:,1]]
    keys=np.minimum(a,b)*partition.count+np.maximum(a,b)
    local=(a>=0)&(b>=0)&((a==b)|np.isin(keys,partition.adjacent_keys));groups=groups[~local]
    np.savez_compressed(cache,lo=lo,hi=hi,face_group=term.face_group,groups=groups,
        material_ids=term._face_material_ids,baseline_pairs=baseline,clearance=np.array(.02),source_sha256=np.array(report['source_sha256']))
    report['frozen_cache']=dict(path=str(cache),sha256=sha(cache));checkpoint('baseline identities and exact boxes cached')
    t=time.perf_counter();optimized,details=ragged(lo,hi,term.face_group,groups,term._face_material_ids,.02,return_report=True);newtime=time.perf_counter()-t
    equal=np.array_equal(optimized,baseline)
    report['ragged_leaf']=dict(**details,seconds=newtime,exact_pair_identity_equal=bool(equal),
        pair_sha256=hashlib.sha256(np.asarray(optimized,dtype='<i8').tobytes()).hexdigest())
    report.update(status='exact_frozen_candidate_equivalence_passed' if equal else 'candidate_identity_mismatch',
        exact_original_predicate_preserved=True,full_baseline_runs=1,unchanged_full_iterations=0,
        conditional_local_partition_only=True,source_state_certified=False)
    checkpoint('done');signal.alarm(0)
    print(json.dumps({'status':report['status'],'baseline_seconds':elapsed,'ragged_leaf_seconds':newtime,'pairs':len(baseline)},indent=2))

if __name__=='__main__':main()
