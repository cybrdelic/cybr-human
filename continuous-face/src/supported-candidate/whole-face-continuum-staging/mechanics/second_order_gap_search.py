"""Bounded source-feasible finite-gap defect correction, SEARCH ONLY.

No force, energy, Hessian, source support, cap or stationarity equation changes.
Current finite-feature normal rows are frozen. All actual trial/path guards and
original-energy Armijo remain authoritative; failed corrections leave plain
search candidates intact. Numerical correction multipliers are discarded.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from exterior_epigraph import normalize_branches
POLICY='stable_interior_feature_gap_defect_search_v1'


def true_max_slope(gradient, data, direction):
    g=np.asarray(gradient,float).ravel();d=np.asarray(direction,float).ravel()
    if data is None:return float(g@d)
    data=normalize_branches(data,len(g));G=csr_matrix(data['branch_gradient_N']);E=np.asarray(data['branch_energy_N_mm']);groups=np.asarray(data['branch_group']);base=np.asarray(data['group_base_branch'])
    value=float((g-np.asarray(G[base].sum(axis=0)).ravel())@d)
    for k in range(len(base)):
        ids=np.flatnonzero(groups==k);win=ids[E[ids].max()-E[ids]<=1e-12]
        value+=float(np.max(G[win]@d))
    return value


def source_metric_correction(factor,A,b):
    """Solve c=W A.T (A W A.T)^-1 b with SOURCE constraints only."""
    A=csr_matrix(A);b=np.asarray(b,float);Z=[];seconds=0.
    if not len(b) or A.shape!=(len(b),factor.n) or not np.isfinite(A.data).all() or not np.isfinite(b).all():
        raise ValueError('Finite positive-rank correction rows required')
    for j in range(len(b)):
        z,_,r=factor.solve(A.getrow(j).toarray().ravel());Z.append(z);seconds+=r['solve_seconds']
    Z=np.column_stack(Z);S=np.asarray(A@Z);diag=np.diag(S)
    if np.any(diag<=0) or not np.isfinite(S).all():raise ValueError('Nonpositive source-metric Schur')
    scale=1/np.sqrt(diag);T=scale[:,None]*S*scale[None,:]
    if np.max(abs(T-T.T))>1e-8:raise ValueError('Nonsymmetric source-metric Schur')
    eigen=np.linalg.eigvalsh((T+T.T)/2)
    if eigen.min()<=1e-12*eigen.max():raise ValueError('Dependent or poorly conditioned correction rows')
    multiplier=scale*np.linalg.solve(T,scale*b);c=Z@multiplier
    source=float(np.max(abs(factor.C@c),initial=0));defect=float(np.max(abs(A@c-b),initial=0))
    if not np.isfinite(c).all() or source>1e-10 or defect>1e-8*np.max(abs(b))+128*np.finfo(float).eps:
        raise ValueError('Original-unit correction/source residual failed')
    return c,dict(source_correction_residual_mm=source,gap_linear_correction_residual_mm=defect,
        scaled_schur_minimum_eigenvalue=float(eigen.min()),source_inverse_seconds=seconds,
        correction_multipliers_are_physical_reactions=False)


def make_gap_anchor(guard,q,factor):
    if factor is None or not hasattr(guard,'interior_curvature_data'):return None
    rows=guard.interior_curvature_data(q)['rows']
    return { (r['site'],r['owner']):r for r in rows
        if r['gap_mm']+guard.penetration_limit<guard.barrier_band }


def gap_corrected_candidate(guard,q,plain,factor,anchor,gradient,branches,*,maximum_rows=8,raw_endpoint=None):
    started=time.perf_counter();info=dict(policy=POLICY,search_only=True,usable_for_stationarity=False,
        source_support_pins_added=False,applied=False,qualified=False)
    if not anchor or factor is None:return None,dict(info,reason='No fresh factor or stable active interior feature')
    if raw_endpoint is None:
        data = guard.interior_curvature_data(plain)
    else:
        from raw_predictor_geometry import cap_only_raw_endpoint, raw_predictor_geometry
        if not cap_only_raw_endpoint(raw_endpoint):
            return None,dict(info,reason='Raw endpoint is not a cap-only diagnostic candidate')
        data = raw_predictor_geometry(guard, plain, anchor)
        info['raw_predictor_geometry'] = data['report']
        if not data['report']['qualified']:
            return None,dict(info,reason=data['report']['reason'])
    trial_rows={(r['site'],r['owner']):r for r in data['rows']}
    move=np.asarray(plain)-q;site_move=np.asarray(guard.C@move);choices=[]
    for key,old in anchor.items():
        new=trial_rows.get(key)
        if new is None or tuple(new['key'])!=tuple(old['key']):continue
        b=float(old['gap_mm']+old['normal']@site_move[old['site']]-new['gap_mm'])
        if np.isfinite(b) and b>0:
            choices.append((b/(old['gap_mm']+guard.penetration_limit),key,old,new,b))
    choices.sort(key=lambda r:r[0],reverse=True);choices=choices[:maximum_rows]
    if not choices:return None,dict(info,reason='No positive same-feature gap defect')
    sites=np.array([r[2]['site'] for r in choices]);normal=np.array([r[2]['normal'] for r in choices]);basis=guard.C[sites].tocoo()
    A=coo_matrix(((basis.data[:,None]*normal[basis.row]).ravel(),(np.repeat(basis.row,3),(3*basis.col[:,None]+np.arange(3)).ravel())),shape=(len(sites),q.size)).tocsr()
    b=np.array([r[4] for r in choices])
    try:c,audit=source_metric_correction(factor,A,b)
    except (ValueError,np.linalg.LinAlgError) as exc:return None,dict(info,reason=str(exc))
    c=c.reshape(q.shape);base=float(np.linalg.norm(site_move,axis=1).max(initial=0));size=float(np.linalg.norm(guard.C@c,axis=1).max(initial=0));corrected=plain+c
    slope=true_max_slope(gradient,branches,corrected-q)
    info.update(applied=True,maximum_base_site_move_mm=base,maximum_site_correction_mm=size,
        correction_to_base_ratio=size/base if base else None,locality_limit_fraction=.1,
        true_max_corrected_displacement_slope_N_mm=slope,**audit,
        rows=[dict(site=k[0],owner=k[1],feature=list(old['key']),gap_defect_mm=float(bb),
            source_faces=list(map(int,old['tied_faces'])),trial_faces=list(map(int,new['tied_faces']))) for _,k,old,new,bb in choices],
        wall_seconds=time.perf_counter()-started)
    if (not base>0 or not size<=.1*base or not np.isfinite([size,base,slope]).all() or not slope<0
            or not np.isfinite(corrected).all() or np.array_equal(corrected,plain)):
        return None,dict(info,reason='Locality, finite, or true max-slope check failed')
    return corrected,dict(info,qualified=True,reason='Bounded source-feasible SEARCH candidate; actual original energy and geometry checks still required')
