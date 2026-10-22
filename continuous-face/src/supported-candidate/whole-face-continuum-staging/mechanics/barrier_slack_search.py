"""Search globalization only: retain half the conservative predicted cap slack.

No energy, derivative, physical cap, accepted-state criterion or history changes.
This module consumes the exact complete owner rows already queried by the guard.
"""
from __future__ import annotations
import numpy as np
POLICY='finite_owner_prediction_relative_half_slack_search_v2'
RETENTION=.5
MINIMUM_PREDICTED_FRACTION=.1

def _rows(guard,q):
    state=guard._state(q)
    if not np.asarray(state['valid'],bool).all() or not np.asarray(state['barrier']['valid'],bool).all():
        raise ValueError('Invalid native owner or barrier rows cannot qualify a search')
    sites=np.asarray(state['force_ids']);owners=np.asarray(state['force_owner']);slack=np.asarray(state['barrier']['slack'],float)
    if sites.ndim!=1 or owners.shape!=sites.shape or slack.shape!=sites.shape or not np.isfinite(slack).all() or np.any(slack<=0):
        raise ValueError('Finite positive shape-matched owner slack rows required')
    keys=[(int(s),int(o)) for s,o in zip(sites,owners)]
    if len(set(keys))!=len(keys) or any(s<0 or o<0 for s,o in keys):
        raise ValueError('Unique valid physical-site/owner identities required')
    return keys,slack,state

def _prediction_anchor(guard,q,state,take):
    """Only actual current closed-owner witnesses can relax the old floor."""
    if not len(take) or not hasattr(guard,'C') or not hasattr(guard,'rigid'):
        return None
    owners=getattr(guard.rigid,'owners',{})
    normals=np.asarray(state.get('normal',[]),float)
    gaps=np.asarray(state.get('force_gap',[]),float)
    q=np.asarray(q,float)
    if (normals.shape!=(len(state['force_ids']),3) or gaps.shape!=(len(normals),)
            or q.ndim!=2 or q.shape!=(guard.C.shape[1],3) or not np.isfinite(q).all()):
        return None
    witnessed=np.array([bool(owners.get(int(state['force_owner'][i]),{}).get('closed',False))
        for i in take]) & np.isfinite(normals[take]).all(1) & np.isfinite(gaps[take]) & (gaps[take]<0)
    # Freeze the current normal and position. Slice only current active rows;
    # later candidates include their entire displacement, including any SOC.
    C=guard.C[np.asarray(state['force_ids'])[take]].tocsr(copy=True)
    return dict(q_mm=q.copy(),C=C,normal=normals[take].copy(),witnessed=witnessed)

def _predicted_slack(trial,anchor):
    old=np.asarray(anchor['slack_mm'],float);reference=old.copy()
    prediction=anchor.get('prediction')
    if prediction is None:return reference,np.zeros(len(old),bool),np.full(len(old),np.nan)
    q=prediction['q_mm'];trial=np.asarray(trial,float)
    if trial.shape!=q.shape or not np.isfinite(trial).all():
        return reference,np.zeros(len(old),bool),np.full(len(old),np.nan)
    C=prediction['C'];normal=prediction['normal'];delta=trial-q
    with np.errstate(invalid='ignore',over='ignore'):
        site_delta=np.asarray(C@delta)
        change=np.einsum('ij,ij->i',normal,site_delta)
        predicted=old+change
        # Floating-point-only allowance, not a geometric slack tolerance:
        # subtraction of represented coordinates, sparse multiply/sum, 3-D dot,
        # slack addition and the 0.1 multiplication. The bound scales with the
        # actual operands and binary64 epsilon, never a fixed distance/ratio.
        eps=np.finfo(float).eps
        operations=np.diff(C.indptr)+6
        gamma=operations*eps/(1-operations*eps)
        magnitude=np.einsum('ij,ij->i',abs(normal),abs(C)@abs(delta))
        coordinate_error=eps*np.einsum('ij,ij->i',abs(normal),abs(C)@(abs(trial)+abs(q)))
        roundoff=coordinate_error+gamma*(old+magnitude)
        floor=MINIMUM_PREDICTED_FRACTION*old
        eligible=(prediction['witnessed'] & np.isfinite(predicted) & np.isfinite(roundoff)
            & (predicted>0) & (change<0) & (predicted+roundoff>=floor))
    # Roundoff at the fraction-to-boundary floor may not lower the floor itself.
    reference[eligible]=np.minimum(old[eligible],np.maximum(predicted[eligible],floor[eligible]))
    return reference,eligible,predicted

def make_slack_anchor(guard,q):
    # Obstacle-free synthetic guards have no native-owner barrier. The actual
    # adopted native factory does, and unknown native contracts fail closed.
    if not hasattr(guard,'_state') or not hasattr(guard,'barrier_band'):
        return None
    if getattr(guard,'provenance',{}).get('query_revision') not in ('complete-band-owner-sum-interior-point-v3', 'complete-band-owner-sum-interior-point-rolling-v4'):
        raise ValueError('Half-slack search requires the qualified complete-owner query contract')
    keys,slack,state=_rows(guard,q)
    take=np.flatnonzero(slack<float(guard.barrier_band))
    return dict(keys=tuple(keys[i] for i in take),slack_mm=slack[take].copy(),barrier_band_mm=float(guard.barrier_band),
                complete_band_released_slack_lower_bound_mm=float(guard.h+guard.penetration_limit),
                prediction=_prediction_anchor(guard,q,state,take))

def check_slack_retention(guard,trial,anchor):
    if anchor is None:
        return dict(policy=POLICY,applied=False,qualified=True,reason='No native-owner cap barrier')
    keys,slack,_=_rows(guard,trial);lookup=dict(zip(keys,slack))
    # A missing old owner is released beyond h by the complete valid finite-band
    # query. An interior owner deeper than the finite band makes _rows invalid,
    # so absence cannot silently excuse penetration or an invalid sign.
    lower=anchor['complete_band_released_slack_lower_bound_mm']
    if not np.isfinite(lower) or lower<=float(guard.barrier_band):
        raise ValueError('Invalid complete-band release lower bound')
    old=dict(zip(anchor['keys'],np.asarray(anchor['slack_mm'],float)))
    predicted_reference,eligible,predicted=_predicted_slack(trial,anchor)
    reference=dict(zip(anchor['keys'],predicted_reference))
    band=anchor['barrier_band_mm']
    # Newly entering/inactive owners start from the band's conservative reference,
    # preventing one feasible trial from entering the barrier at microscopic slack.
    checked=tuple(sorted(set(old)|{k for k,s in lookup.items() if s<band}))
    before=np.array([reference.get(k,band) for k in checked],float)
    current=np.array([old.get(k,band) for k in checked],float)
    after=np.array([lookup.get(k,lower) for k in checked],float)
    if not np.isfinite(before).all() or np.any(before<=0):raise ValueError('Invalid anchor')
    okay=after>=RETENTION*before
    bad=np.flatnonzero(~okay)
    return dict(policy=POLICY,applied=True,qualified=bool(okay.all()),retained_fraction=RETENTION,
        checked_owner_terms=len(before),released_owner_terms=sum(k not in lookup for k in checked),
        newly_active_owner_terms=sum(k not in old for k in checked),
        prediction_relative_owner_terms=int(eligible.sum()),
        minimum_predicted_fraction=MINIMUM_PREDICTED_FRACTION,
        prediction_uses_actual_candidate_displacement_including_SOC=True,
        minimum_trial_to_current_slack_ratio=None if not len(current) else float(np.min(after/current)),
        minimum_trial_to_reference_slack_ratio=None if not len(before) else float(np.min(after/before)),
        offenders=[dict(site=checked[i][0],owner=checked[i][1],
            reference_slack_mm=float(before[i]),trial_slack_mm=float(after[i])) for i in bad[:8]])
