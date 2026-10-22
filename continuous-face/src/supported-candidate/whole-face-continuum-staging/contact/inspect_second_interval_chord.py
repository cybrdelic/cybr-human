"""One preserved 5-to-10-ms failed physical interval, no dynamics."""
from pathlib import Path
import json,time
import numpy as np
from coarse_physical_site_contact_epigraph import create_guard
from coarse_physical_site_contact import sha
from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle
from coarse_rigid_query import _bins

HERE=Path(__file__).resolve().parent
started=time.perf_counter()
solid=load_geometry_with_source_fibres(HERE.parent/'geometry/whole-face-continuum-final.npz',HERE.parent/'geometry/whole-face-muscle-phases-final.npz')
supports,_=load_support_bundle(solid,HERE.parent/'mechanics/coarse-physical-site-v1/supports')
guard=create_guard(solid,supports)
source=HERE.parent/'mechanics/coarse-barrier-v1/sequence-step2-v2/passive-step0002-NONPHYSICAL-checkpoint.npz'
packet=np.load(source);q0=packet['q_previous_mm'];q1=packet['nonphysical_iterate_mm']
prior=HERE/'temporal-audits/v8-passive-step1-accepted.npz'
with np.load(prior) as z:
    assert np.array_equal(q0,z['positions_mm'][-1])
    assert np.array_equal(packet['velocity_previous_mm_s'],z['velocities_mm_s'][-1])
report_path=source.parent/'report.json';run=json.loads(report_path.read_text())['runs']['passive']['reports'][-1]
count=max(1,int(np.ceil(guard.max_physical_site_difference(q0,q1)/guard.path_spacing)))
alpha_failure=(len(run['frame_path']['samples'])-1)/count
state=guard._state(q0+alpha_failure*(q1-q0));native=guard.rigid.native
bad_sites=np.unique(np.r_[state['ids'][~state['valid']],state['force_ids'][state['force_gap']<=-guard.penetration_limit]])
points=state['points'][bad_sites]
inside,valid=native.containment(points)
# Independent generic ray direction for only these points, using the unchanged
# all-closed-owner finite-ray kernel and a rebuilt rotated index.
direction=np.array([.573,.231,1.]);direction/=np.linalg.norm(direction)
e1=np.cross(direction,[0.,1.,0.]);e1/=np.linalg.norm(e1);e2=np.cross(direction,e1)
matrix=np.column_stack((e1,e2,direction));tri=native.ray_triangles@matrix
bounds=[]
for owner in native.owner_ids:
    t=native.triangles[native.owners[int(owner)]['ids']]@matrix
    bounds.append((t.min((0,1)),t.max((0,1))))
oblique,oblique_valid=native._ray(points@matrix,tri,_bins(tri,2),bounds=bounds)
rows=[]
for j,site in enumerate(bad_sites):
    point=points[j]
    owners=native.owner_ids[inside[j]]
    query_index=np.flatnonzero(state['ids']==site)[0]
    history=[]
    for alpha in sorted(set((0.,alpha_failure/2,alpha_failure,(alpha_failure+1)/2,1.))):
        p=(guard.C[site]@(q0+alpha*(q1-q0))).reshape(1,3)
        interior,ok=native.containment(p)
        owner_rows=[]
        for owner in owners:
            faces=native.owners[int(owner)]['ids']
            witness,d2=native._finite_witness(np.broadcast_to(p[0],(len(faces),3)),faces)
            win=np.lexsort((faces,d2))[0];distance=float(np.sqrt(d2[win]))
            isin=bool(interior[0,np.searchsorted(native.owner_ids,owner)])
            owner_rows.append(dict(owner=int(owner),closed=bool(native.owners[int(owner)]['closed']),
                inside=isin,unsigned_distance_mm=distance,signed_gap_mm=(-distance if isin else distance),
                closest_filtered_face=int(faces[win]),closest_point_mm=witness[win].tolist(),
                triangle_vertex_ids=native.faces[faces[win]].tolist(),all_owner_facets_enumerated=len(faces)))
        history.append(dict(alpha=alpha,point_mm=p[0].tolist(),containment_valid=bool(ok[0]),owners=owner_rows))
    row=dict(site=int(site),native_query_valid=bool(state['valid'][query_index]),
        query_global_gap_mm=float(state['gap'][query_index]),reference_point_mm=guard.P0[site].tolist(),
        interval_start_point_mm=(guard.C[site]@q0)[0].tolist(),
        independent_oblique_ray_valid=bool(oblique_valid[j]),
        independent_oblique_owner_membership_matches=bool(np.array_equal(inside[j],oblique[j])),history=history)
    rows.append(row);print(json.dumps(row),flush=True)
report=dict(scope='One actual preserved chord failure; complete finite witnesses of implicated owners only',
    checkpoint_path=str(source),checkpoint_sha256=sha(source),report_sha256=sha(report_path),
    prior_accepted_npz_sha256=sha(prior),actual_prior_q_v_match=True,
    physical_interval_s=[.005,.01],alpha_failure=alpha_failure,failed_sites=len(bad_sites),
    native_sign_queries_valid=bool(valid.all()),independent_oblique_all_valid=bool(oblique_valid.all()),
    independent_oblique_membership_matches=bool(np.array_equal(inside,oblique)),sites=rows,
    total_seconds=time.perf_counter()-started)
(HERE/'second-interval-chord-witnesses.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sites'}),flush=True)
