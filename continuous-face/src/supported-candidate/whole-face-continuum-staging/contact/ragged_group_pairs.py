"""Exact bounded-memory leaf expansion for already-qualified coarse group pairs.

No new exclusion, distance approximation or collision predicate is introduced.
Every unordered fine pair in each supplied coarse pair is visited once. Integer
prefix sums flatten ragged Cartesian products into bounded vectorized batches.
The final AABB test is literally the original outward predicate, with the same
ascending fine-face order; this also preserves asymmetric floating rounding.
"""
import numpy as np
import time


def candidates(lo,hi,face_group,groups,material_ids,clearance=0.,*,batch_size=262144,return_report=False):
    lo=np.asarray(lo,float);hi=np.asarray(hi,float);face_group=np.asarray(face_group,int)
    groups=np.asarray(groups,int).reshape(-1,2);material_ids=np.asarray(material_ids,int)
    if not np.isfinite(clearance) or clearance<0 or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi<lo):
        raise ValueError('invalid bounds/clearance')
    if batch_size<=0:raise ValueError('positive bounded batch required')
    if lo.shape!=hi.shape or lo.shape!=(len(face_group),3) or material_ids.shape!=(len(face_group),3):
        raise ValueError('shape mismatch')
    count=int(face_group.max(initial=-1))+1
    if len(groups) and (np.any(groups<0) or np.any(groups>=count) or np.any(groups[:,0]>=groups[:,1])):
        raise ValueError('canonical distinct group pairs required')
    order=np.argsort(face_group,kind='stable');counts=np.bincount(face_group,minlength=count)
    offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    sizes=counts[groups[:,0]]*counts[groups[:,1]]
    prefix=np.r_[0,np.cumsum(sizes,dtype=np.int64)]
    total=int(prefix[-1]);out=[];aabb_hits=0;visited=0;batches=0
    expansion_seconds=0.;aabb_seconds=0.;shared_seconds=0.;axis_survivors=np.zeros(3,np.int64)
    for start in range(0,total,batch_size):
        tick=time.perf_counter()
        stop=min(total,start+batch_size);flat=np.arange(start,stop,dtype=np.int64)
        pair=np.searchsorted(prefix,flat,side='right')-1
        local=flat-prefix[pair];g0,g1=groups[pair,0],groups[pair,1]
        a=order[offsets[g0]+local//counts[g1]]
        b=order[offsets[g1]+local%counts[g1]]
        a,b=np.minimum(a,b),np.maximum(a,b)
        expansion_seconds+=time.perf_counter()-tick;tick=time.perf_counter()
        # Identical conjunction, evaluated axis by axis so early rejects do not
        # allocate/read the remaining coordinate arrays. No epsilon changes.
        for axis in range(3):
            ok=(np.nextafter(hi[a,axis]+clearance,np.inf)>=lo[b,axis])&(np.nextafter(lo[a,axis]-clearance,-np.inf)<=hi[b,axis])
            a,b=a[ok],b[ok];axis_survivors[axis]+=len(a)
        aabb_hits+=len(a);aabb_seconds+=time.perf_counter()-tick;tick=time.perf_counter()
        if len(a):
            shared=(material_ids[a,:,None]==material_ids[b,None,:]).any((1,2))
            if np.any(~shared):out.append(np.c_[a[~shared],b[~shared]])
        shared_seconds+=time.perf_counter()-tick
        visited+=stop-start;batches+=1
    tick=time.perf_counter()
    result=np.concatenate(out) if out else np.empty((0,2),int)
    if len(result):result=result[np.lexsort((result[:,1],result[:,0]))]
    report=dict(strategy='bounded ragged parent-pair Cartesian expansion with original exact AABB and material filters',
        cartesian_pairs_visited=visited,batches=batches,batch_size=batch_size,aabb_hits=aabb_hits,
        leaf_candidate_pairs=len(result),coarse_pairs=len(groups),approximate_exclusions=0,
        expansion_seconds=expansion_seconds,aabb_seconds=aabb_seconds,shared_filter_seconds=shared_seconds,
        final_sort_seconds=time.perf_counter()-tick,axis_survivors=axis_survivors.tolist())
    return (result,report) if return_report else result
