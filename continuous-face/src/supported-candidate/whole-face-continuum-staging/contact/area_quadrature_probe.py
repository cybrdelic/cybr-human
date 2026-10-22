"""Diagnostic-only reciprocal reference-area rule, NOT an adopted guard/law.

Fixed positive lumped reference-vertex areas integrate barrier density. Distances
are to the nearest eligible FINITE target triangle, with both facet directions
queried and multiplicity deduplicated. Every finite triangle geometry/path guard
remains separate. Vertex quadrature can under-resolve localized contact and this
probe does not establish physical connectivity or pressure calibration.
"""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
# Import initializes the unchanged qualified contact-module paths.
import continuum_contact
from moving_triangle_contact import point_triangle_geometry,activated_log_barrier,InfeasibleContactError


def reference_vertex_areas(C,faces,X):
    V=np.asarray(C@X);faces=np.asarray(faces,int)
    triangle=np.linalg.norm(np.cross(V[faces[:,1]]-V[faces[:,0]],V[faces[:,2]]-V[faces[:,0]]),axis=1)/2
    if np.any(triangle<=0) or not np.isfinite(triangle).all():raise ValueError('positive finite reference triangle areas required')
    area=np.zeros(len(V))
    for slot in range(3):np.add.at(area,faces[:,slot],triangle/3)
    return area


def evaluate_area_rule(C,faces,q,reference_areas,pairs,clearance=.02,coefficient=.001):
    faces=np.asarray(faces,int);pairs=np.asarray(pairs,int).reshape(-1,2);area=np.asarray(reference_areas,float)
    V=np.asarray(C@q)
    if area.shape!=(len(V),) or np.any(area<0) or not np.isfinite(area).all() or not np.isfinite(coefficient) or coefficient<=0:
        raise ValueError('fixed nonnegative areas and explicit positive N/mm^3 coefficient required')
    if not len(pairs):return 0.,np.zeros_like(q),dict(active_vertices=0,unique_vertex_target_queries=0)
    # Both orientations; repeated incident-source faces must never multiply the
    # quadrature measure. Each (physical vertex,target facet) query is unique.
    source=np.r_[faces[pairs[:,0]].ravel(),faces[pairs[:,1]].ravel()]
    target=np.r_[np.repeat(pairs[:,1],3),np.repeat(pairs[:,0],3)]
    encoded=np.unique(source.astype(np.int64)*len(faces)+target)
    source=encoded//len(faces);target=encoded%len(faces)
    gap=np.empty(len(source));normal=np.empty((len(source),3));bary=np.empty_like(normal)
    for lo in range(0,len(source),65536):
        hi=min(lo+65536,len(source));geo=point_triangle_geometry(V[source[lo:hi]],V[faces[target[lo:hi]]])
        gap[lo:hi]=geo['gap'];normal[lo:hi]=geo['normal'];bary[lo:hi]=geo['barycentric']
    order=np.lexsort((target,gap,source));first=np.r_[True,source[order[1:]]!=source[order[:-1]]];pick=order[first]
    if np.any(gap[pick]<=0) or not np.isfinite(gap[pick]).all():raise InfeasibleContactError('area probe sees zero/nonfinite finite distance')
    pick=pick[gap[pick]<clearance];v=source[pick];t=target[pick]
    b,db,_=activated_log_barrier(gap[pick],clearance);weights=.5*coefficient*area[v]
    rows=np.repeat(np.arange(len(v)),3)
    target_carrier=csr_matrix((bary[pick].ravel(),(rows,faces[t].ravel())),shape=(len(v),C.shape[0]))@C
    D=(C[v]-target_carrier).tocsr()
    gradient=np.asarray(D.T@((weights*db)[:,None]*normal[pick]))
    energy=float(weights@b)
    report=dict(active_vertices=len(v),unique_vertex_target_queries=len(source),
        candidate_facet_pairs=len(pairs),coefficient_N_per_mm3=float(coefficient),
        reciprocal_factor=.5,active_reference_area_mm2=float(area[v].sum()),
        minimum_active_gap_mm=None if not len(v) else float(gap[pick].min()),
        total_gradient_N=np.sum(gradient,axis=0).tolist(),
        energy_density_barrier_units='mm^2',reference_area_units='mm^2',
        active_vertex_ids=v,active_target_faces=t,active_gap_mm=gap[pick],
        active_reference_area_mm2_values=area[v],active_energy_N_mm=weights*b,
        active_carrier_difference=D,active_point_force_factor_N=weights*db,
        active_normal=normal[pick],
        quadrature='fixed lumped reference vertex area; nearest finite target per vertex; reciprocal half measure',
        geometric_guards_replaced=False,physical_connectivity_inferred=False,
        contact_pressure_calibrated=False,rule_adopted=False)
    return energy,gradient,report
