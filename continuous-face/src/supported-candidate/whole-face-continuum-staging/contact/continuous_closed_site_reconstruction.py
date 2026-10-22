"""Declared inter-frame reconstruction allowance, not physical contact softening.

The physical endpoint cap remains 0.05 mm. The fixed 0.005 mm reconstruction
allowance applies only to the affine saved-frame chord, certified at 0.055 mm.
This is an explicit temporal reconstruction approximation, not surface CCD.

Uses existing native sign/distance queries only. Clipped signed owner distance
is 1-Lipschitz. Endpoint lower slacks certify an interval if their sum exceeds
its length. A shared finite triangle's convex p-neighborhood additionally
certifies near-cap tangential intervals without microscopic subdivision.
Open unsigned sheets cannot violate this penetration cap; their non-crossing is
NOT certified. Existing positive-J and sampled path guards remain mandatory.
"""
from pathlib import Path
import time
import numpy as np
from coarse_physical_site_contact import sha


from continuous_closed_site_path import _triangle_tube


def certify_closed_owner_reconstruction_chord(guard,q0,q1,*,max_midpoint_queries=200000,max_depth=24,enable_triangle_tubes=True):
    started=time.perf_counter()
    q0,q1=np.asarray(q0,float),np.asarray(q1,float)
    if q0.shape!=q1.shape or q0.ndim!=2 or q0.shape[1]!=3 or not np.isfinite(q0).all() or not np.isfinite(q1).all():
        raise ValueError('Finite matching N x 3 affine carrier endpoints required')
    if max_midpoint_queries<0 or max_depth<0:raise ValueError('Nonnegative work bounds required')
    native=getattr(guard.rigid,'native',guard.rigid)
    P0,P1=guard.C@q0,guard.C@q1
    h,endpoint_cap=guard.h,guard.penetration_limit
    if endpoint_cap!=.05:raise ValueError("This fixed reconstruction contract requires the unchanged 0.05 mm endpoint cap")
    allowance=.005
    p=endpoint_cap+allowance
    closed=np.unique(np.asarray([o for o in native.owner_ids if native.owners[int(o)]['closed']],int))
    report=dict(feasible=False,continuous_closed_owner_site_cap_checked=False,
        scope='Affine occupied-site reconstruction chords versus closed-owner 0.055 mm cap; physical endpoints remain at 0.05 mm',
        physical_endpoint_penetration_cap_mm=endpoint_cap,interframe_reconstruction_allowance_mm=allowance,
        certified_interframe_penetration_cap_mm=p,physical_contact_law_changed=False,
        global_surface_collision_certified=False,unsigned_open_sheet_non_crossing_certified=False,
        positive_J_path_must_be_checked_separately=True,
        native_sign_semantics='Unchanged independently valid closed-owner finite ray classification',
        numerical_scope='Conservative padded floating-point finite-feature bounds, not exact-predicate CAD',
        penetration_cap_mm=p,physical_sites=len(P0),closed_owners=len(closed),
        midpoint_queries=0,triangle_tube_owner_interval_certificates=0,maximum_refinement_depth=0,
        source_sha256=sha(Path(__file__)),max_midpoint_queries=max_midpoint_queries,max_depth=max_depth)
    def finish(reason,good=False,**extra):
        return dict(report,**extra,feasible=good,continuous_closed_owner_site_cap_checked=good,
            reason=reason,wall_seconds=time.perf_counter()-started)
    scale=np.max(abs(native.vertices),initial=0)
    def pad(points):return 2048*np.finfo(float).eps*(np.linalg.norm(points,axis=1)+scale+h+p+1)
    def matrix(local_sites,owner,gaps,count,default=None):
        out=(np.full((count,len(closed)),h) if default is None
             else np.broadcast_to(default[:,None],(count,len(closed))).copy())
        ci=np.searchsorted(closed,owner)
        keep=ci<len(closed)
        keep[keep]&=closed[ci[keep]]==owner[keep]
        out[local_sites[keep],ci[keep]]=gaps[keep]
        return out
    states=[guard._state(q0),guard._state(q1)]
    global_g=[];defaults=[]
    for s,points in zip(states,(P0,P1)):
        if not s['valid'].all():return finish('Endpoint native query invalid')
        gap=np.full(len(P0),h);gap[s['ids']]=s['gap']
        if not np.isfinite(gap).all() or np.any(gap+endpoint_cap<=0):return finish('Endpoint violates unchanged physical site penetration cap')
        # Existing reference/rolling anchors are exterior to EVERY closed
        # owner. Their translated positive unsigned bound is also a lower
        # bound on every unclipped signed gap. Keep this stronger far-field
        # information rather than forcing long chords down to the h cap.
        bound=np.full(len(points),h)
        if hasattr(guard,'rolling'):
            bound=np.maximum(bound,guard.rolling.lower_bound(points))
        elif hasattr(guard,'lower') and hasattr(guard,'P0'):
            bound=np.maximum(bound,guard.lower-np.linalg.norm(points-guard.P0,axis=1))
        if bound.shape!=(len(points),) or not np.isfinite(bound).all():
            raise ValueError('Finite matching exterior anchor bounds required')
        known=bound[s['ids']]
        contradiction=(s['gap']<h)&(known>h)&(known>s['gap']+pad(points[s['ids']]))
        if np.any(contradiction):
            return finish('Exterior anchor contradicts already queried native contact; not safe',
                inconsistent_anchor_sites=s['ids'][contradiction][:16].tolist())
        gap=np.maximum(gap,np.where(bound>h,bound,-np.inf))
        defaults.append(bound);global_g.append(gap)
    if not len(closed):return finish('No closed owners; open-sheet non-crossing is not certified',True)
    length=np.linalg.norm(P1-P0,axis=1)
    pad0,pad1=pad(P0),pad(P1)
    good=(global_g[0]+p-pad0)+(global_g[1]+p-pad1)>length+pad0+pad1
    ids=np.flatnonzero(~good)
    report['initial_global_interval_certificates']=int(good.sum())
    if not len(ids):return finish('All site intervals have positive Lipschitz lower slack',True)
    gaps=[]
    for s,default in zip(states,defaults):
        loc=np.searchsorted(ids,s['force_ids'])
        keep=loc<len(ids);keep[keep]&=ids[loc[keep]]==s['force_ids'][keep]
        gaps.append(matrix(loc[keep],s['force_owner'][keep],s['force_gap'][keep],len(ids),default[ids]))
    a,b=np.zeros(len(ids)),np.ones(len(ids));g0,g1=gaps
    for depth in range(max_depth+1):
        report['maximum_refinement_depth']=depth
        left=P0[ids]+a[:,None]*(P1[ids]-P0[ids])
        right=P0[ids]+b[:,None]*(P1[ids]-P0[ids])
        pl,pr=pad(left),pad(right)
        segment=(b-a)*length[ids]+pl+pr
        owner_good=(g0+p-pl[:,None])+(g1+p-pr[:,None])>segment[:,None]
        if enable_triangle_tubes:
            rr,cc=np.nonzero(~owner_good & (abs(g0)<p) & (abs(g1)<p))
            for row,col in zip(rr,cc):
                witness=_triangle_tube(native,left[row],right[row],closed[col],p,max(pl[row],pr[row]))
                if witness is not None:
                    owner_good[row,col]=True
                    report['triangle_tube_owner_interval_certificates']+=1
        uncertain=~owner_good.all(1)
        if not uncertain.any():return finish('Every closed-owner interval has a positive cap-slack certificate',True)
        ids,a,b,g0,g1=[x[uncertain] for x in (ids,a,b,g0,g1)]
        if depth==max_depth or report['midpoint_queries']+len(ids)>max_midpoint_queries:
            return finish('Inconclusive refinement work bound; not safe',unresolved_intervals=len(ids))
        mid=(a+b)/2
        points=P0[ids]+mid[:,None]*(P1[ids]-P0[ids])
        raw=native.query_band(points,clearance=h,penetration_limit=p,owner_terms=True)
        report['midpoint_queries']+=len(ids)
        invalid=~raw['valid'] | ~np.isfinite(raw['signed_gap']) | (raw['signed_gap']+p<=0)
        if invalid.any():
            ii=np.flatnonzero(invalid)[:16]
            offenders=[dict(site=int(ids[j]),alpha=float(mid[j]),point_mm=points[j].tolist(),
                clipped_query_gap_mm=float(raw['signed_gap'][j]),native_query_valid=bool(raw['valid'][j]),
                penetration_limit_exceeded=bool(raw['penetration_limit_exceeded'][j])) for j in ii]
            return finish('Midpoint native query invalid or declared reconstruction penetration cap violated',offenders=offenders)
        gm=matrix(raw['contact_site_index'],raw['contact_owner_id'],raw['contact_signed_gap'],len(ids))
        olda,oldb=a,b
        ids=np.r_[ids,ids];a=np.r_[olda,mid];b=np.r_[mid,oldb]
        g0,g1=np.r_[g0,gm],np.r_[gm,g1]
    raise AssertionError('Unreachable refinement state')
