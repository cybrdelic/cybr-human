"""Bounded saved-state query/branch qualification; does not run dynamics."""
from pathlib import Path
import json
import time
import numpy as np
from coarse_physical_site_contact_epigraph import create_guard
from coarse_physical_site_contact import sha
from source_transfer import load_geometry_with_source_fibres
from clipped_origin_support import load_support_bundle

HERE=Path(__file__).resolve().parent
started=time.perf_counter()
solid=load_geometry_with_source_fibres(HERE.parent/'geometry/whole-face-continuum-final.npz',HERE.parent/'geometry/whole-face-muscle-phases-final.npz')
supports,_=load_support_bundle(solid,HERE.parent/'mechanics/coarse-physical-site-v1/supports')
guard=create_guard(solid,supports)
source=HERE.parent/'mechanics/coarse-barrier-v1/shared-checkpoint-inspection/coupled-direction.npz'
packet=np.load(source);q=packet['positions_mm'];direction=packet['direction_mm']
events=json.loads((HERE/'saved-contact-feature-events.json').read_text())['events']
rows=[]
for alpha in (0.,.03125):
    x=q+alpha*direction
    before=time.perf_counter();bundle=guard.contact_branches(x);wall=time.perf_counter()-before
    state=guard._state(x);bases=bundle['group_base_branch']
    chosen=np.asarray(bundle['branch_gradient_N'][bases].sum(0)).reshape(x.shape)
    gradient_error=float(np.max(abs(bundle['smooth_contact_gradient_N']+chosen-state['gradient'])))
    energies=bundle['branch_energy_N_mm'];groups=bundle['branch_group']
    maximum=np.full(len(bases),-np.inf);np.maximum.at(maximum,groups,energies)
    assert np.array_equal(maximum,energies[bases])
    assert gradient_error<1e-15
    energy_error=abs(bundle['smooth_contact_energy_N_mm']+maximum.sum()-state['energy'])
    assert energy_error<1e-12
    measured=[]
    for event in events:
        site,owner=event['site'],event['owner']
        g=np.flatnonzero((bundle['group_site']==site)&(bundle['group_owner']==owner))
        assert len(g)==1
        branch=np.flatnonzero(groups==g[0]);faces=bundle['branch_triangle'][branch]
        expected=[f['global_face'] for f in event['features']]
        assert all(face in faces for face in expected)
        # Independent all-triangles-of-owner check at this exact actual point.
        native=guard.rigid.native;owner_faces=native.owners[owner]['ids']
        point=state['points'][site]
        witness,d2=native._finite_witness(np.broadcast_to(point,(len(owner_faces),3)),owner_faces)
        inside=owner_faces[np.sqrt(d2)<guard.h]
        represented=set(sum([bundle['branch_equivalent_native_faces'][j] for j in branch],[]))
        assert represented==set(inside.tolist())
        measured.append(dict(site=site,owner=owner,branches=len(branch),winner=int(bundle['branch_triangle'][bases[g[0]]]),
            all_native_owner_finite_features_checked=len(owner_faces),all_positive_features_retained=len(inside),
            measured_cusp_faces=expected,measured_cusp_energies_N_mm=[float(energies[branch[np.flatnonzero(faces==f)[0]]]) for f in expected]))
    row=dict(alpha=alpha,groups=len(bases),branches=len(energies),gradient_reconstruction_max_error_N=gradient_error,
        energy_reconstruction_error_N_mm=energy_error,all_group_winners_match_exact_max=True,
        wall_seconds=wall,branch_construction_seconds=bundle['provenance']['construction_seconds'],measured_cusps=measured)
    rows.append(row);print(json.dumps(row),flush=True)
report=dict(scope='Saved actual states and finite branch queries only, no dynamics',qualified=True,
    original_source_sha256=sha(source),adapter_sha256=sha(HERE/'coarse_physical_site_contact_epigraph.py'),
    provider_sha256=sha(HERE/'exterior_feature_bundles.py'),frozen_base_sha256=sha(HERE/'coarse_physical_site_contact.py'),
    frozen_native_helper_sha256=sha(HERE/'coarse_rigid_query.py'),rows=rows,total_seconds=time.perf_counter()-started)
(HERE/'exterior-feature-bundles-qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
