"""One preserved actual-state tangent check, no solve or coordinate writes."""
from pathlib import Path
import json
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent; W=HERE.parent.parent
for p in (W/'mechanics',HERE.parent,HERE):sys.path.insert(0,str(p))
from coarse_physical_site_contact import sha
from coarse_physical_site_contact_epigraph import create_guard as create_old
from coarse_physical_site_contact_curvature import InteriorCurvaturePhysicalContact
from interior_feature_curvature import assemble_interior_curvature
from compact_penetration_barrier import compact_barrier_terms
from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle

started=time.perf_counter()
checkpoint=W/'mechanics/coarse-barrier-v1/automatic-reconstruction-020-v1/recovery-001-checkpoint-at-chunk1/passive-step0007-NONPHYSICAL-checkpoint.npz'
untouched=[checkpoint]+[HERE.parent/name for name in ('coarse_physical_site_contact.py','coarse_physical_site_contact_epigraph.py','coarse_physical_site_contact_rolling.py','coarse_rigid_query.py','compact_penetration_barrier.py','exterior_feature_bundles.py','continuous_closed_site_path.py')]
hashes={str(p):sha(p) for p in untouched}
packet=np.load(checkpoint,allow_pickle=False);q=packet['nonphysical_iterate_mm']
print(json.dumps(dict(stage='loading_unchanged_contact',checkpoint_sha256=sha(checkpoint))),flush=True)
solid=load_geometry_with_source_fibres(W/'geometry/whole-face-continuum-final.npz',W/'geometry/whole-face-muscle-phases-final.npz')
supports,_=load_support_bundle(solid,W/'mechanics/coarse-physical-site-v1/supports')
old=create_old(solid,supports);old.set_barrier_scale(.2)
state=old._state(q)
new=InteriorCurvaturePhysicalContact.from_epigraph(old)
e,g=old.energy(q);ne,ng=new.energy(q)
assert e==ne and np.array_equal(g,ng)
assert all(new.provenance[k]==v for k,v in old.provenance.items())
assert old.endpoint(q)==new.endpoint(q)
for name in ('energy','_state','endpoint','path','safe_step_bound','contact_branches','contact_accounting'):
    assert getattr(type(old),name) is getattr(type(new),name)
data=new.interior_curvature_data(q)
print(json.dumps(dict(stage='all_saved_interior_terms_classified',**data['report'])),flush=True)
rows=[r for r in data['rows'] if r['site']==319342 and r['owner']==5]
assert len(rows)==1
row=rows[0];assert row['kind']=='edge';assert set(row['tied_faces'])=={34348,34753}
assert set(row['key'][1:])=={17317,17257}
Hadd=assemble_interior_curvature(new,data);H0=old.tangent(q);H1=new.tangent(q)
assert np.max(abs((H1-H0-Hadd).data),initial=0)<2e-6
native=new.rigid.native;site=row['site'];owner=row['owner'];p=state['points'][site];t=row['basis'][0]
mu=new.barrier_scale*new.k*new.weights[site]*(new.h+new.penetration_limit)*new.barrier_band
# Local exact owner law uses the same native complete query and original terms.
def local(point):
    query=native.query_band(np.asarray(point)[None],clearance=new.h,penetration_limit=new.penetration_limit,owner_terms=True)
    assert query['valid'].all()
    ix=np.flatnonzero(query['contact_owner_id']==owner);assert len(ix)==1;i=ix[0]
    gap=query['contact_signed_gap'][i];normal=query['contact_normal'][i]
    b=compact_barrier_terms([gap],mu,penetration_limit=new.penetration_limit,band=new.barrier_band)
    assert b['valid'].all()
    R=new.k*new.weights[site]*(new.h-gap)+b['reaction'][0]
    return gap,-R*normal
fd=[]
for eps in (1e-6,3e-7,1e-7):
    gp,dp=local(p+eps*t);gm,dm=local(p-eps*t)
    estimate=float(t@((dp-dm)/(2*eps)))
    relative=abs(estimate-row['stiffness_N_per_mm'])/row['stiffness_N_per_mm']
    fd.append(dict(epsilon_mm=eps,tangent_quadratic_form_fd_N_per_mm=estimate,
        predicted_added_N_per_mm=row['stiffness_N_per_mm'],relative_error=relative,
        minimum_slack_mm=min(gp,gm)+new.penetration_limit,
        signed_gap_second_difference_per_mm=(gp+gm-2*row['gap_mm'])/eps**2))
assert fd[-1]['relative_error']<1e-4
assert abs(np.dot(row['normal'],t))<1e-14 and abs(np.dot(row['edge_unit'],t))<1e-14
# Independent all-owner-face enumeration certifies complete nearest ties at hotspot.
owner_faces=native.owners[owner]['ids'];ww,dd=native._finite_witness(np.broadcast_to(p,(len(owner_faces),3)),owner_faces)
minimum=np.sqrt(dd).min();pad=2048*np.finfo(float).eps*(np.linalg.norm(p)+np.max(abs(native.vertices))+new.h+1)
ties=owner_faces[np.sqrt(dd)<=minimum+pad]
assert set(ties)==set(row['tied_faces'])
# Single hotspot assembly and non-unit original C weights, no duplicate faces.
single=assemble_interior_curvature(new,dict(rows=[row]));indices=new.C[site].indices;basis=new.C[site].data
expected=row['stiffness_N_per_mm']*np.outer(t,t)
block=single[3*5395:3*5395+3,3*5395:3*5395+3].toarray()
nodeweight=float(new.C[site,5395]);np.testing.assert_allclose(block,nodeweight**2*expected,rtol=1e-13,atol=1e-10)
assert all(sha(p)==h for p,h in hashes.items())
summary={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in row.items()}
report=dict(qualified=True,scope='One actual preserved nonphysical q; no dynamics, new solve, or accepted-state edits',
    checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),barrier_scale=.2,
    energy_gradient_bit_identical=True,endpoint_report_identical=True,inherited_guard_methods_identical=True,
    original_provenance_exact_subset=True,original_sources_and_checkpoint_unchanged=True,
    original_source_hashes=hashes,interior_feature_tangent_source_sha256=sha(HERE/'coarse_physical_site_contact_curvature.py'),
    interior_feature_curvature_source_sha256=sha(HERE/'interior_feature_curvature.py'),
    complete_saved_state_feature_report=data['report'],hotspot=summary,directional_finite_difference=fd,
    original_normal_stiffness_N_per_mm=float(new.k*new.weights[site]+state['barrier']['curvature'][row['force_row']]),
    owner_native_triangles_independently_checked=len(owner_faces),node5395_basis_weight=nodeweight,
    node5395_added_tangential_stiffness_N_per_mm=nodeweight**2*row['stiffness_N_per_mm'],
    added_tangent_nonzeros=Hadd.nnz,wall_seconds=time.perf_counter()-started)
(HERE/'saved-hotspot-qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
